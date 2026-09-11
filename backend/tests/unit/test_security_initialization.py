"""部署初始化只操作临时项目；验证密钥隔离、私有权限和重复执行保护。"""

from pathlib import Path
import os

from dotenv import dotenv_values
import pytest

from scripts.init_security import initialize_security_config


@pytest.fixture
def project(tmp_path):
    template = Path(__file__).resolve().parents[3] / ".env.example"
    (tmp_path / ".env.example").write_bytes(template.read_bytes())
    return tmp_path


def test_production_configuration_uses_distinct_secrets_and_private_files(project):
    env_path, signing_path = initialize_security_config(project, profile="production")
    values = dotenv_values(env_path)
    signing_key = signing_path.read_text(encoding="utf-8").strip()
    assert len({signing_key, values["REDIS_PASSWORD"], values["MINIO_ROOT_PASSWORD"]}) == 3
    assert min(len(value) for value in (signing_key, values["REDIS_PASSWORD"], values["MINIO_ROOT_PASSWORD"])) >= 43
    assert values["RATE_LIMIT_STORAGE"] == f"redis://:{values['REDIS_PASSWORD']}@127.0.0.1:6379/0"
    assert values["ALLOWED_ORIGINS"] == "https://arxiv.001769.xyz"
    assert values["AUTH_MODE"] == "jwt" and values["ALLOW_PUBLIC_REGISTRATION"] == "false"
    assert values["JWT_SECRET_KEY"] == "" and values["BACKEND_API_KEYS"] == ""
    assert values["JWT_SECRET_FILE"] == "backend/config/production.jwt-secret"
    assert signing_key not in env_path.read_text(encoding="utf-8")
    assert values["TRUSTED_PROXY_IPS"] == "127.0.0.1,::1"
    if os.name != "nt":
        assert env_path.stat().st_mode & 0o777 == 0o600
        assert signing_path.stat().st_mode & 0o777 == 0o600


def test_development_and_production_do_not_share_signing_keys_or_accounts(project):
    dev_env, dev_key = initialize_security_config(project, profile="development")
    prod_env, prod_key = initialize_security_config(project, profile="production")
    development, production = dotenv_values(dev_env), dotenv_values(prod_env)
    assert development["RATE_LIMIT_STORAGE"] == "memory://"
    assert development["TRUSTED_PROXY_IPS"] == ""
    assert development["AUTH_DATABASE_PATH"] != production["AUTH_DATABASE_PATH"]
    assert dev_key.read_bytes() != prod_key.read_bytes()


@pytest.mark.parametrize("existing", [".env.production", "backend/config/production.jwt-secret"])
def test_initialization_never_overwrites_existing_credentials(project, existing):
    path = project / existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"existing-deployment-content")
    with pytest.raises(FileExistsError):
        initialize_security_config(project, profile="production")
    assert path.read_bytes() == b"existing-deployment-content"
    missing = "backend/config/production.jwt-secret" if existing == ".env.production" else ".env.production"
    assert not (project / missing).exists()


@pytest.mark.parametrize("domain", ["https://arxiv.001769.xyz", "example.com/path", "example.com\nAUTH_MODE=api_key", "-bad.example.com"])
def test_invalid_domain_fails_before_any_secret_is_written(project, domain):
    with pytest.raises(ValueError):
        initialize_security_config(project, profile="production", domain=domain)
    assert not (project / ".env.production").exists()
    assert not (project / "backend/config/production.jwt-secret").exists()


def test_security_runner_isolates_starlette_and_dotenv_file_readers(monkeypatch, tmp_path):
    import dotenv
    from starlette.config import Config
    from scripts.test_security import isolate_configuration_reads

    runtime = tmp_path / "run"
    runtime.mkdir()
    private_config = tmp_path / "private.env"
    test_config = runtime / "cli.env"
    private_config.write_text("LOADER_PROBE=outside-test-runtime\n", encoding="utf-8")
    test_config.write_text("LOADER_PROBE=isolated-test-value\n", encoding="utf-8")
    monkeypatch.delenv("LOADER_PROBE", raising=False)
    # 保存父 runner 的隔离层，当前测试结束后恢复，不能污染后续测试的配置读取。
    monkeypatch.setattr(dotenv, "load_dotenv", dotenv.load_dotenv)
    monkeypatch.setattr(Config, "_read_file", Config._read_file)
    isolate_configuration_reads(runtime)
    assert Config(private_config)("LOADER_PROBE", default="absent") == "absent"
    assert not dotenv.load_dotenv(private_config)
    assert Config(test_config)("LOADER_PROBE") == "isolated-test-value"
