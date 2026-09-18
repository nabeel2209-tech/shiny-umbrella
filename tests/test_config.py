from trading.core.config import Settings


def make(**kw):
    return Settings(_env_file=None, **kw)


def test_defaults_are_paper_and_not_live():
    s = make()
    assert s.broker == "paper"
    assert s.live_trading is False
    assert s.live_orders_allowed is False
    assert s.feature_marketplace is False


def test_live_requires_dhan():
    s = make(live_trading=True, broker="paper")
    assert s.live_orders_allowed is False
    assert any("LIVE_TRADING" in p for p in s.problems())


def test_dhan_requires_credentials():
    s = make(broker="dhan")
    assert any("DHAN_CLIENT_ID" in p for p in s.problems())
    s = make(broker="dhan", dhan_client_id="1", dhan_access_token="supersecret", live_trading=True)
    assert s.live_orders_allowed is True
    assert not any("DHAN" in p for p in s.problems())
    assert s.dhan_access_token.get_secret_value() == "supersecret"
    assert "supersecret" not in repr(s)  # secrets never leak into logs


def test_empty_redis_url_is_none():
    assert make(redis_url="").redis_url is None
