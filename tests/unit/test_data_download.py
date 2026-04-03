from __future__ import annotations

import unittest
from datetime import UTC, datetime

from hybrid_quant.data import BinanceHistoricalDownloader, DownloadRequest


def _kline_row(open_ms: int, open_price: float) -> list[object]:
    return [
        open_ms,
        str(open_price),
        str(open_price + 1.0),
        str(open_price - 1.0),
        str(open_price + 0.5),
        "10.0",
        open_ms + 299999,
        "100.0",
        50,
        "5.0",
        "50.0",
        "0",
    ]


class _FakeResponse:
    def __init__(self, payload: list[list[object]]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[list[object]]:
        return self._payload


class _FakeSession:
    def __init__(self, payloads: list[list[list[object]]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, *, params: dict[str, object], timeout: int) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return _FakeResponse(self.payloads.pop(0))


class DataDownloadTests(unittest.TestCase):
    def test_binance_downloader_paginates_and_builds_dataframe(self) -> None:
        session = _FakeSession(
            payloads=[
                [
                    _kline_row(1704067200000, 100.0),
                    _kline_row(1704067500000, 101.0),
                ],
                [
                    _kline_row(1704067800000, 102.0),
                ],
            ]
        )
        downloader = BinanceHistoricalDownloader(base_url="https://example.test", session=session)
        request = DownloadRequest(
            symbol="BTCUSDT",
            interval="5m",
            start=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            limit=2,
        )

        frame = downloader.download(request)

        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.index.name, "open_time")
        self.assertEqual(frame.iloc[0]["symbol"], "BTCUSDT")
        self.assertEqual(frame.iloc[-1]["interval"], "5m")
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0]["params"]["startTime"], 1704067200000)
        self.assertEqual(session.calls[1]["params"]["startTime"], 1704067800000)

