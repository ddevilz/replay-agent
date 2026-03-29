from __future__ import annotations


from replay.strategies.redaction import FieldRedaction, NoOpRedaction, PIIRedaction


class TestNoOpRedaction:
    def test_returns_data_unchanged(self) -> None:
        data = {"key": "value", "nested": {"a": 1}}
        result = NoOpRedaction().redact(data)
        assert result == data

    def test_returns_same_reference(self) -> None:
        data = {"k": "v"}
        assert NoOpRedaction().redact(data) is data


class TestFieldRedaction:
    def test_redacts_top_level_field(self) -> None:
        data = {"api_key": "sk-secret", "prompt": "hello"}
        result = FieldRedaction(["api_key"]).redact(data)
        assert result["api_key"] == "***REDACTED***"
        assert result["prompt"] == "hello"

    def test_redacts_nested_field(self) -> None:
        data = {"outer": {"api_key": "secret", "safe": "value"}}
        result = FieldRedaction(["api_key"]).redact(data)
        assert result["outer"]["api_key"] == "***REDACTED***"
        assert result["outer"]["safe"] == "value"

    def test_redacts_field_inside_list(self) -> None:
        data = {"items": [{"token": "abc"}, {"token": "xyz"}]}
        result = FieldRedaction(["token"]).redact(data)
        assert result["items"][0]["token"] == "***REDACTED***"
        assert result["items"][1]["token"] == "***REDACTED***"

    def test_non_targeted_fields_pass_through(self) -> None:
        data = {"safe": "data", "also_safe": 42}
        result = FieldRedaction(["api_key"]).redact(data)
        assert result == data

    def test_multiple_fields(self) -> None:
        data = {"api_key": "k", "password": "p", "username": "u"}
        result = FieldRedaction(["api_key", "password"]).redact(data)
        assert result["api_key"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"
        assert result["username"] == "u"


class TestPIIRedaction:
    def test_redacts_openai_key(self) -> None:
        data = {"msg": "my key is sk-abcdefghijklmnopqrstuvwxyz123456"}
        result = PIIRedaction().redact(data)
        assert "sk-" not in result["msg"]
        assert "OPENAI_KEY" in result["msg"]

    def test_redacts_email(self) -> None:
        data = {"user": "contact me at user@example.com"}
        result = PIIRedaction().redact(data)
        assert "user@example.com" not in result["user"]
        assert "EMAIL" in result["user"]

    def test_redacts_nested_pii(self) -> None:
        data = {"outer": {"inner": "email is test@test.com"}}
        result = PIIRedaction().redact(data)
        assert "test@test.com" not in result["outer"]["inner"]

    def test_non_sensitive_strings_pass_through(self) -> None:
        data = {"greeting": "hello world", "count": 42}
        result = PIIRedaction().redact(data)
        assert result["greeting"] == "hello world"
        assert result["count"] == 42
