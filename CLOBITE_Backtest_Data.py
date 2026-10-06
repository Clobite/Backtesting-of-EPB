# -*- coding: utf-8 -*-
"""CLOBITE: 시총 2,000억 이상 종목, KRX/NXT 08:00~10:00 3분봉.
python CLOBITE_Backtest_Data.py [--date YYYY-MM-DD] [--output intraday]
매매/주문 기능 없음. Python 3.11+, requests pandas numpy 필요.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://openapi.koreainvestment.com:9443"
MIN_MARKET_CAP = 200_000_000_000
NAVER_API_BASE = "https://stock.naver.com"
NAVER_PAGE_SIZE = 100
NAVER_MAX_PAGES_PER_MARKET = 30
NAVER_SLEEP = 0.10
KST = ZoneInfo("Asia/Seoul")
def to_num(x):
    try:
        if x is None:
            return np.nan
        s = str(x).replace(",", "").replace("%", "").strip()
        if s in ["", "nan", "None", "N/A"]:
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def normalize_code(x):
    s = str(x).strip()
    if s.endswith(".KS") or s.endswith(".KQ"):
        s = s[:-3]
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        s = s.replace(".0", "")
    if s.isdigit():
        return s.zfill(6)
    return s

def is_excluded_name(name):
    """
    ETF/ETN/스팩/리츠 등 제외.
    우선주는 제외하지 않음.
    """
    n = str(name).upper()

    exclude_keywords = [
        "KODEX", "TIGER", "ACE", "SOL", "RISE", "KBSTAR", "HANARO",
        "KOSEF", "ARIRANG", "TIMEFOLIO", "ETF", "ETN",
        "PLUS", "KIWOOM", "KOACT", "WON", "HK", "1Q",
        "스팩", "SPAC", "리츠", "REIT"
    ]

    return any(k.upper() in n for k in exclude_keywords)

def _naver_api_get(path, params=None):
    """
    현재 네이버 증권(stock.naver.com) 공개 화면이 사용하는 읽기 전용 JSON API 호출.
    - 비공식/미문서화 내부 API이므로 응답 구조가 바뀔 수 있어 방어적으로 처리한다.
    - 403/429는 무리한 재시도를 하지 않고 즉시 실패 처리한다.
    """
    url = f"{NAVER_API_BASE}{path}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://stock.naver.com/market/stock/kr/stocklist/capitalization",
    }

    last_error = None

    for attempt in range(1, 4):
        try:
            res = requests.get(url, headers=headers, params=params, timeout=20)

            if res.status_code in (403, 429):
                raise RuntimeError(
                    f"네이버 증권 API 접근 제한 HTTP {res.status_code}: {res.url}"
                )

            res.raise_for_status()

            content_type = (res.headers.get("content-type") or "").lower()
            if "json" not in content_type and not res.text.lstrip().startswith(("[", "{")):
                raise RuntimeError(
                    f"네이버 증권 API가 JSON이 아닌 응답을 반환했습니다. "
                    f"status={res.status_code}, content-type={content_type}"
                )

            return res.json()

        except RuntimeError:
            raise
        except Exception as e:
            last_error = e
            if attempt < 3:
                time.sleep(0.7 * attempt)
            else:
                break

    raise RuntimeError(f"네이버 증권 API 호출 실패: {last_error}")

def _extract_stock_rows(payload):
    """
    네이버 API 응답에서 종목 row 리스트를 최대한 유연하게 추출한다.
    현재 default endpoint는 리스트를 직접 반환할 수 있고,
    향후 items/content/stocks/result 등 래퍼가 생겨도 대응한다.
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if not isinstance(payload, dict):
        return []

    preferred_keys = [
        "items", "content", "stocks", "stockList", "list", "result", "data"
    ]

    for key in preferred_keys:
        value = payload.get(key)

        if isinstance(value, list):
            dict_rows = [x for x in value if isinstance(x, dict)]
            if dict_rows:
                return dict_rows

        if isinstance(value, dict):
            nested = _extract_stock_rows(value)
            if nested:
                return nested

    # 마지막 방어: dict 내부의 리스트 중 item code 형태를 가진 리스트 탐색
    for value in payload.values():
        if isinstance(value, list):
            rows = [x for x in value if isinstance(x, dict)]
            if any(
                any(k in r for k in ("itemCode", "itemcode", "code", "symbolCode"))
                for r in rows
            ):
                return rows

    return []

def _first_nonempty(row, keys, default=None):
    for key in keys:
        if key in row:
            value = row.get(key)
            if value not in [None, "", "null", "None", "nan"]:
                return value
    return default

def _parse_market_cap_won(value):
    """
    네이버 시가총액 값을 원 단위로 통일한다.

    대응 형식:
    - '450조 1,234억'
    - '1조'
    - '3,250억'
    - 4,501,234 (억원 단위 숫자)
    - 450123400000000 (원 단위 숫자)

    신규 API marketSum은 현재 시총 랭킹에서 사용되는 값이며,
    숫자만 오는 경우 한국 주식 시총 규모를 기준으로
    1e11 미만은 억원 단위로 해석한다.
    """
    if value is None:
        return np.nan

    s = str(value).strip().replace(",", "")
    if s in ["", "nan", "None", "null", "-"]:
        return np.nan

    # 한글 단위 표기
    jo = 0.0
    eok = 0.0

    m = re.search(r"([0-9.]+)\s*조", s)
    if m:
        jo = float(m.group(1))

    m = re.search(r"([0-9.]+)\s*억", s)
    if m:
        eok = float(m.group(1))

    if jo or eok:
        return jo * 1_000_000_000_000 + eok * 100_000_000

    # 숫자형
    cleaned = re.sub(r"[^0-9.\-]", "", s)
    if cleaned in ["", "-", "."]:
        return np.nan

    try:
        num = float(cleaned)
    except Exception:
        return np.nan

    # 원 단위 시총이면 그대로, 그보다 작으면 억원 단위로 해석
    if abs(num) >= 100_000_000_000:
        return num

    return num * 100_000_000

def _normalize_naver_stock_row(row, market):
    code = _first_nonempty(
        row,
        ["itemCode", "itemcode", "code", "symbolCode", "stockCode"]
    )
    name = _first_nonempty(
        row,
        ["itemName", "itemname", "name", "stockName", "stockNameKor"]
    )
    market_cap_raw = _first_nonempty(
        row,
        ["marketSum", "marketCap", "marketValue", "marketCapitalization"]
    )

    if code is None or name is None or market_cap_raw is None:
        return None

    code = normalize_code(code)
    name = str(name).strip()
    market_cap = _parse_market_cap_won(market_cap_raw)

    if not code or not name or pd.isna(market_cap):
        return None

    return {
        "Code": code,
        "종목명": name,
        "Market": market,
        "Industry": "",
        "시가총액": float(market_cap),
        "네이버시총_억원": float(market_cap) / 100_000_000,
    }

def fetch_naver_market_sum_page(market: str, start_idx: int = 0, page_size: int = NAVER_PAGE_SIZE) -> pd.DataFrame:
    """
    신규 네이버 증권 시가총액 순위 API.

    현재 공개 페이지에서 확인된 endpoint:
    /api/domestic/market/stock/default
      ?tradeType=KRX
      &marketType=KOSPI|KOSDAQ
      &orderType=marketSum
      &startIdx=...
      &pageSize=...

    HTML 파싱(pd.read_html)을 사용하지 않는다.
    """
    if market not in ["KOSPI", "KOSDAQ"]:
        raise ValueError(f"지원하지 않는 market: {market}")

    payload = _naver_api_get(
        "/api/domestic/market/stock/default",
        params={
            "tradeType": "KRX",
            "marketType": market,
            "orderType": "marketSum",
            "startIdx": int(start_idx),
            "pageSize": int(page_size),
        },
    )

    rows = _extract_stock_rows(payload)
    normalized = []

    for row in rows:
        item = _normalize_naver_stock_row(row, market)
        if item is not None:
            normalized.append(item)

    if not normalized:
        return pd.DataFrame(
            columns=["Code", "종목명", "Market", "Industry", "시가총액", "네이버시총_억원"]
        )

    return pd.DataFrame(normalized)

def load_naver_cap_targets() -> pd.DataFrame:
    """
    KOSPI/KOSDAQ 시가총액 순으로 신규 네이버 증권 JSON API를 조회한다.

    최적화:
    - pageSize=100
    - 시총 내림차순이므로 한 페이지 전체가 2,000억 미만이면 해당 시장 조회 종료
    - 중복 페이지가 반복되면 종료해 무한루프 방지
    """
    rows = []

    for market in ["KOSPI", "KOSDAQ"]:
        print(f"[INFO] 네이버 신규 API {market} 시총 순위 수집 시작")

        seen_codes = set()
        start_idx = 0
        paging_mode = None  # None -> 1-based next-page 테스트, 필요 시 offset으로 전환
        previous_page_codes = set()

        for page_no in range(1, NAVER_MAX_PAGES_PER_MARKET + 1):
            d = fetch_naver_market_sum_page(
                market=market,
                start_idx=start_idx,
                page_size=NAVER_PAGE_SIZE,
            )

            if d.empty:
                print(f"[INFO] {market} API page {page_no}: 데이터 없음 → 종료")
                break

            page_codes = set(d["Code"].astype(str))
            new_codes = page_codes - seen_codes

            # startIdx 의미가 페이지 index인지 row offset인지 자동 감지
            if page_no == 2 and paging_mode is None and previous_page_codes:
                overlap_ratio = len(page_codes & previous_page_codes) / max(len(page_codes), 1)

                if overlap_ratio >= 0.80:
                    # startIdx=1이 첫 페이지와 거의 같으면 row offset 방식으로 간주하고
                    # 두 번째 페이지를 pageSize offset으로 다시 조회한다.
                    print(
                        f"[INFO] {market} startIdx가 row offset 방식으로 감지됨 "
                        f"(중복률 {overlap_ratio:.1%}) → offset={NAVER_PAGE_SIZE} 재조회"
                    )
                    paging_mode = "offset"
                    start_idx = NAVER_PAGE_SIZE
                    d = fetch_naver_market_sum_page(
                        market=market,
                        start_idx=start_idx,
                        page_size=NAVER_PAGE_SIZE,
                    )
                    if d.empty:
                        break
                    page_codes = set(d["Code"].astype(str))
                    new_codes = page_codes - seen_codes
                else:
                    paging_mode = "page"

            if not new_codes:
                print(f"[WARN] {market} API page {page_no}: 신규 종목 0개 → 반복 페이지로 판단, 종료")
                break

            d = d[d["Code"].astype(str).isin(new_codes)].copy()
            rows.append(d)
            seen_codes.update(new_codes)

            min_cap = pd.to_numeric(d["시가총액"], errors="coerce").min()
            max_cap = pd.to_numeric(d["시가총액"], errors="coerce").max()

            print(
                f"[INFO] {market} API page {page_no}: {len(d)}개 신규 / "
                f"누적 {len(seen_codes)}개 / "
                f"시총 {max_cap / 1e8:,.0f}억 ~ {min_cap / 1e8:,.0f}억"
            )

            # 시총 내림차순이므로 페이지의 최대 시총마저 기준 미만이면 종료 가능
            if pd.notna(max_cap) and max_cap < MIN_MARKET_CAP:
                print(f"[INFO] {market}: 시총 2,000억 미만 구간 진입 → 조회 종료")
                break

            # 다음 startIdx
            if paging_mode == "offset":
                start_idx += NAVER_PAGE_SIZE
            else:
                start_idx += 1

            previous_page_codes = page_codes
            time.sleep(NAVER_SLEEP)

    if not rows:
        raise RuntimeError(
            "네이버 신규 증권 API 시총 후보 수집 결과가 비어 있습니다. "
            "stock.naver.com API 구조 변경 여부를 확인하세요."
        )

    master = pd.concat(rows, ignore_index=True)
    master = master.drop_duplicates(subset=["Code"], keep="first")

    before = len(master)
    master = master[~master["종목명"].apply(is_excluded_name)].copy()
    print(
        f"[INFO] ETF/ETN/스팩/리츠 제외 후: {len(master):,}개 / "
        f"제외 {before - len(master):,}개"
    )

    master["시가총액"] = pd.to_numeric(master["시가총액"], errors="coerce")
    master = master[master["시가총액"] >= MIN_MARKET_CAP].copy()
    master = master.sort_values("시가총액", ascending=False).reset_index(drop=True)

    print(f"[INFO] 네이버 신규 API 기준 시총 2,000억 이상 최종 대상: {len(master):,}개")

    if master.empty:
        raise RuntimeError(
            "네이버 API 응답은 받았지만 시총 2,000억 이상 종목이 0개입니다. "
            "marketSum 단위 또는 응답 필드 변경 여부를 확인하세요."
        )

    return master[
        ["Code", "종목명", "Market", "Industry", "시가총액", "네이버시총_억원"]
    ].copy()

class KIS:
    def __init__(self):
        self.key = os.environ['KIS_APP_KEY']
        self.secret = os.environ['KIS_APP_SECRET']
        self.session = requests.Session()
        r = self.session.post(BASE_URL + '/oauth2/tokenP', json={
            'grant_type': 'client_credentials', 'appkey': self.key,
            'appsecret': self.secret}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if not payload.get('access_token'):
            raise RuntimeError('KIS 인증 실패 (키/토큰은 로그에 출력하지 않습니다)')
        self.headers = {'authorization': 'Bearer ' + payload['access_token'],
                        'appkey': self.key, 'appsecret': self.secret,
                        'custtype': 'P'}

    def safe_text(self, value):
        text = str(value)
        for secret in (self.key, self.secret, self.headers.get('authorization', '')):
            if secret:
                text = text.replace(secret, '[REDACTED]')
        return text.replace('\n', ' ')[:500]

    def get(self, path, tr, params):
        context = (f"TR={tr} market={params.get('FID_COND_MRKT_DIV_CODE')} "
                   f"code={params.get('FID_INPUT_ISCD')} "
                   f"date={params.get('FID_INPUT_DATE_1', '')} "
                   f"time={params.get('FID_INPUT_HOUR_1', '')}")
        last_error = 'UNKNOWN'
        for attempt in range(4):
            time.sleep(0.15)
            retryable = True
            try:
                r = self.session.get(BASE_URL + path, params=params,
                    headers={**self.headers, 'tr_id': tr}, timeout=30)
                if r.status_code != 200:
                    retryable = r.status_code == 429 or r.status_code >= 500
                    last_error = f'HTTP {r.status_code}'
                    try:
                        body = r.json()
                        last_error += ' ' + self.safe_text(body.get('msg_cd', '')) + ' ' + self.safe_text(body.get('msg1', ''))
                    except ValueError:
                        pass
                    raise RuntimeError(last_error)
                data = r.json()
                if str(data.get('rt_cd')) != '0':
                    msg_cd = str(data.get('msg_cd', 'UNKNOWN'))
                    last_error = (f"rt_cd={data.get('rt_cd')} msg_cd={msg_cd} "
                                  f"msg1={self.safe_text(data.get('msg1', ''))}")
                    retryable = msg_cd in {'EGW00201', 'EGW00123'}
                    raise RuntimeError(last_error)
                return data
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                if isinstance(exc, requests.RequestException):
                    last_error = type(exc).__name__  # 요청 헤더/키를 출력하지 않음
                elif isinstance(exc, ValueError):
                    last_error = '응답 JSON 해석 실패'
                print(f'[API ERROR] {context} attempt={attempt+1}/4 {last_error}', flush=True)
                if not retryable or attempt == 3:
                    raise RuntimeError(f'{context}: {last_error}') from None
                time.sleep(2 ** attempt)

    def daily(self, code, market, date):
        end = date - dt.timedelta(days=1)
        return self.get('/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice',
            'FHKST03010100', {'FID_COND_MRKT_DIV_CODE': market,
            'FID_INPUT_ISCD': code, 'FID_INPUT_DATE_1': (end-dt.timedelta(days=60)).strftime('%Y%m%d'),
            'FID_INPUT_DATE_2': end.strftime('%Y%m%d'), 'FID_PERIOD_DIV_CODE': 'D',
            'FID_ORG_ADJ_PRC': '1'}).get('output2', [])

    def minutes(self, code, market, date):
        cursor = dt.datetime.combine(date, dt.time(9,59,59))
        lower = dt.datetime.combine(date, dt.time(9 if market == 'J' else 8))
        rows, seen = [], set()
        for page in range(30):
            data = self.get('/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice',
                'FHKST03010230', {'FID_COND_MRKT_DIV_CODE': market,
                'FID_INPUT_ISCD': code, 'FID_INPUT_DATE_1': date.strftime('%Y%m%d'),
                'FID_INPUT_HOUR_1': cursor.strftime('%H%M%S'),
                'FID_PW_DATA_INCU_YN': 'N', 'FID_FAKE_TICK_INCU_YN': 'N'})
            batch = data.get('output2', [])
            if not batch:
                return rows
            stamps = []
            added = 0
            for row in batch:
                stamp = dt.datetime.strptime(str(row['stck_bsop_date']) + str(row['stck_cntg_hour']).zfill(6), '%Y%m%d%H%M%S')
                stamps.append(stamp)
                if stamp > cursor:
                    raise RuntimeError(f'API 시간 경계 오류: 요청={cursor}, 반환={stamp}')
                if stamp not in seen:
                    seen.add(stamp)
                    rows.append({**row, '_stamp': stamp})
                    added += 1
            oldest = min(stamps)
            # 시장 시작 봉 도달 시 종료: 시작 전 조회가 첫 봉으로 보정되는 응답 방지.
            # 첫 봉 거래대금 기준이 없으면 TradingValue는 빈칸으로 유지.
            if oldest <= lower:
                return rows
            if not added or oldest >= cursor:
                raise RuntimeError('분봉 페이지 진행 정체')
            cursor = oldest - dt.timedelta(seconds=1)
        raise RuntimeError('분봉 페이지 상한 도달: 수집 완료 여부 불명')


def prior_close(rows, prev_date):
    selected = [r for r in rows if r.get('stck_bsop_date') == prev_date]
    if not selected:
        return None
    value = to_num(selected[0].get('stck_clpr'))
    return value if pd.notna(value) and value > 0 else None


def aggregate(rows, date):
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).sort_values('_stamp').drop_duplicates('_stamp')
    # KIS 분봉 체결시각을 분의 시작 라벨로 취급. 첫 실계좌에서 확인 필요.
    d['_minute'] = pd.to_datetime(d['_stamp']).dt.floor('min')
    if d['_minute'].duplicated().any():
        raise RuntimeError('동일 분에 여러 봉: 분봉 응답 규격 확인 필요')
    start = pd.Timestamp(date).replace(hour=8)
    # 수집 범위 밖 응답은 OHLC 검증에서 제외. 직전 1분은 거래대금 기준만 사용.
    d = d[(d['_minute'] >= start-pd.Timedelta(minutes=1)) &
          (d['_minute'] < start+pd.Timedelta(hours=2))].copy()
    if d.empty:
        return pd.DataFrame()
    for target, source in [('Open','stck_oprc'),('High','stck_hgpr'),
        ('Low','stck_lwpr'),('Close','stck_prpr'),('Volume','cntg_vol'),
        ('CumTradingValue','acml_tr_pbmn')]:
        d[target] = d[source].map(to_num) if source in d else np.nan
    if d[['Open','High','Low','Close','Volume']].isna().any().any():
        missing = [c for c in ['stck_oprc','stck_hgpr','stck_lwpr','stck_prpr','cntg_vol'] if c not in d or d[c].map(to_num).isna().any()]
        raise RuntimeError(f'OHLCV 누락: {missing}; 응답 필드={list(d.columns)}')
    if ((d['High'] < d[['Open','Close','Low']].max(axis=1)) |
        (d['Low'] > d[['Open','Close','High']].min(axis=1)) |
        (d['Low'] <= 0) | (d['Volume'] < 0)).any():
        sample = d[['Open','High','Low','Close','Volume']].head(2).to_dict('records')
        raise RuntimeError(f'OHLCV 값 오류: sample={sample}')
    delta = d['CumTradingValue'].diff()
    contiguous = d['_minute'].diff() == pd.Timedelta(minutes=1)
    d['TradingValue'] = delta.where(contiguous & (delta >= 0))
    start = pd.Timestamp(date).replace(hour=8)
    d = d[(d['_minute'] >= start) & (d['_minute'] < start + pd.Timedelta(hours=2))].copy()
    if d.empty:
        return pd.DataFrame()
    d['_bar'] = d['_minute'].dt.floor('3min')
    result = d.groupby('_bar', sort=True).agg(Open=('Open','first'),
        High=('High','max'), Low=('Low','min'), Close=('Close','last'),
        Volume=('Volume','sum'), CumTradingValue=('CumTradingValue','last'),
        MinuteCount=('_minute','count'))
    result['TradingValue'] = d.groupby('_bar')['TradingValue'].agg(
        lambda s: s.sum() if s.notna().all() else np.nan)
    result['Time'] = result.index.strftime('%H:%M:%S')
    result['EndTime'] = (result.index + pd.Timedelta(minutes=3)).strftime('%H:%M:%S')
    result['BarStatus'] = np.where(result['MinuteCount']==3, 'OBSERVED_3_MINUTES', 'SPARSE_MINUTES')
    return result.reset_index(drop=True)


def atomic_write(path, content):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(content)
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--date', default=dt.datetime.now(KST).date().isoformat())
    p.add_argument('--output', default='intraday')
    args = p.parse_args()
    date = dt.date.fromisoformat(args.date)
    now = dt.datetime.now(KST)
    if date > now.date() or (date == now.date() and now.time() < dt.time(10,5)):
        raise RuntimeError('전체 08~10시 수집은 해당일 10:05 KST 이후 실행하세요')
    if date.weekday() >= 5:
        print('주말: latest 유지')
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    api = KIS()
    benchmark = api.daily('005930', 'J', date)
    days = sorted({r['stck_bsop_date'] for r in benchmark
                   if r.get('stck_bsop_date') and to_num(r.get('stck_clpr')) > 0})
    if not days:
        raise RuntimeError('직전 거래일 확인 실패')
    prev_date = days[-1]
    # 오늘/휴장일은 실데이터 유무로 판정. 빈 응답을 NXT 비대상으로 단정하지 않음.
    universe = load_naver_cap_targets()
    frames, errors, reports = [], [], []
    collected = dt.datetime.now(KST).isoformat(timespec='seconds')
    consecutive_errors = 0
    tag = date.strftime('%Y%m%d')
    for i, target in universe.iterrows():
        code = str(target['Code'])
        stage = 'KRX 전일종가'
        try:
            krx_close = prior_close(api.daily(code, 'J', date), prev_date)
            stage = 'NXT 전일종가'
            nxt_close = prior_close(api.daily(code, 'NX', date), prev_date)
            for market, label in [('J','KRX'),('NX','NXT')]:
                stage = f'{label} 분봉 조회/3분봉 집계'
                bars = aggregate(api.minutes(code, market, date), date)
                status = 'OK' if not bars.empty else 'NO_DATA_UNCONFIRMED'
                reports.append({'Code':code,'TradingMarket':label,'Status':status,'Bars':len(bars)})
                if bars.empty:
                    bars = pd.DataFrame([{'Time':None, 'EndTime':None, 'BarStatus':status}])
                for key,value in {'Date':date.isoformat(),'Code':code,'종목명':target['종목명'],
                    'Market':target['Market'],'TradingMarket':label,
                    'UniverseAsOf':collected,'MarketCap':target['시가총액'],
                    'PrevDate':dt.datetime.strptime(prev_date,'%Y%m%d').date().isoformat(),
                    'PrevCloseKRX':krx_close,'PrevCloseNXT':nxt_close,
                    'PrevCloseKRXStatus':'CONFIRMED' if krx_close is not None else 'MISSING',
                    'PrevCloseNXTStatus':'CONFIRMED' if nxt_close is not None else 'NO_DATA_UNCONFIRMED',
                    'CollectedAt':collected,'Source':'KIS','DataStatus':status}.items():
                    bars[key] = value
                frames.append(bars)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            error = {'Code':code,'종목명':target['종목명'],'Stage':stage,'Error':api.safe_text(exc)}
            errors.append(error)
            print(f'[ERROR] {code} {stage}: {error["Error"]}', flush=True)
            atomic_write(out/f'KIS_3min_{tag}_report.json',
                json.dumps({'Date':date.isoformat(),'Errors':errors,'Markets':reports,
                            'Status':'COLLECTION_IN_PROGRESS_OR_ABORTED'},ensure_ascii=False,indent=2).encode('utf-8'))
            if consecutive_errors >= 3:
                raise RuntimeError('3종목 연속 실패: 조기 중단. 위 [ERROR]/[API ERROR] 및 report 확인') from None
        print(f'[{i+1}/{len(universe)}] {code}: errors={len(errors)}', flush=True)
    tag = date.strftime('%Y%m%d')
    metadata = {'Date':date.isoformat(),'PrevDate':prev_date,'UniverseCount':len(universe),
                'UniverseAsOf':collected,'Errors':errors,'Markets':reports,
                'Window':'[08:00:00,10:00:00) Asia/Seoul',
                'Note':'NO_DATA_UNCONFIRMED does not prove NXT eligibility or suspension.'}
    atomic_write(out/f'KIS_3min_{tag}_report.json',
                 json.dumps(metadata,ensure_ascii=False,indent=2).encode('utf-8'))
    if errors:
        raise RuntimeError(f'{len(errors)}종목 수집 오류: 날짜 CSV/latest 모두 유지. report 확인')
    result = pd.concat(frames,ignore_index=True)
    if not (result['DataStatus']=='OK').any():
        print('전 시장 실데이터 없음 (휴장 또는 데이터 부재): latest 유지')
        return
    cols = ['Date','Code','종목명','Market','TradingMarket','Time','EndTime',
        'Open','High','Low','Close','Volume','TradingValue','CumTradingValue',
        'MinuteCount','BarStatus','PrevDate','PrevCloseKRX','PrevCloseNXT',
        'PrevCloseKRXStatus','PrevCloseNXTStatus','MarketCap','UniverseAsOf',
        'Source','DataStatus','CollectedAt']
    result = result.reindex(columns=cols).sort_values(['Code','TradingMarket','Time'])
    content = result.to_csv(index=False,na_rep='').encode('utf-8-sig')
    # 過去日 재실행은 날짜 파일만 저장: 최신 거래일을 과거로 되돌리지 않음.
    dated, latest = out/f'KIS_3min_{tag}.csv', out/'KIS_3min_latest.csv'
    update_latest = True
    if latest.exists():
        old_dates = pd.read_csv(latest,usecols=['Date'])['Date'].dropna().unique()
        if len(old_dates) != 1:
            raise RuntimeError('기존 latest 날짜가 혼합되어 있습니다')
        update_latest = not len(old_dates) or str(old_dates[0]) <= date.isoformat()
    atomic_write(dated,content)
    if update_latest:
        atomic_write(latest,content)
        assert dated.read_bytes() == latest.read_bytes()
    print(f'DONE: {dated}; latest updated={update_latest}; rows={len(result)}')

if __name__ == '__main__':
    main()
