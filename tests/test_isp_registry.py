# tests/test_isp_registry.py
"""Tests del ISP Registry — validar que el registro es consistente y correcto."""

from __future__ import annotations

import pytest

from src.scrapers.isp_registry import (
    ISPConfig,
    ISP_REGISTRY,
    ScrapingMode,
    brand_to_legal_name,
    get_all_isps,
    get_isp_config,
)


class TestISPRegistryLookup:
    """Tests de búsqueda y acceso al registro."""

    def test_get_known_isp_returns_config(self):
        config = get_isp_config("netlife")
        assert isinstance(config, ISPConfig)
        assert config.brand == "Netlife"
        assert config.legal_name == "MEGADATOS S.A."

    def test_get_unknown_isp_raises_key_error(self):
        with pytest.raises(KeyError, match="isp_inexistente"):
            get_isp_config("isp_inexistente")

    def test_all_isps_returns_list_of_keys(self):
        keys = get_all_isps()
        assert isinstance(keys, list)
        assert len(keys) >= 7
        assert "netlife" in keys
        assert "claro" in keys

    def test_registry_has_expected_isps(self):
        expected = {"netlife", "claro", "xtrim", "cnt", "ecuanet", "fibramax", "alfanet", "celerity"}
        assert expected.issubset(set(ISP_REGISTRY.keys()))


class TestISPConfigIntegrity:
    """Tests de integridad de datos de cada ISPConfig."""

    @pytest.mark.parametrize("isp_key", list(ISP_REGISTRY.keys()))
    def test_every_isp_has_brand_and_legal_name(self, isp_key: str):
        config = get_isp_config(isp_key)
        assert config.brand, f"{isp_key}: brand no puede estar vacío"
        assert config.legal_name, f"{isp_key}: legal_name no puede estar vacío"

    @pytest.mark.parametrize("isp_key", list(ISP_REGISTRY.keys()))
    def test_every_isp_has_plans_url(self, isp_key: str):
        config = get_isp_config(isp_key)
        assert config.plans_url.startswith("https://"), \
            f"{isp_key}: plans_url debe comenzar con https://"

    @pytest.mark.parametrize("isp_key", list(ISP_REGISTRY.keys()))
    def test_scraping_mode_is_valid_enum(self, isp_key: str):
        config = get_isp_config(isp_key)
        assert isinstance(config.scraping_mode, ScrapingMode)


class TestCompanyLookupHelper:
    """Tests del helper de búsqueda de empresa."""

    def test_brand_to_legal_name_found(self):
        result = brand_to_legal_name("Netlife")
        assert result == "MEGADATOS S.A."

    def test_brand_to_legal_name_case_insensitive(self):
        result = brand_to_legal_name("NETLIFE")
        assert result == "MEGADATOS S.A."

    def test_brand_to_legal_name_unknown_returns_brand(self):
        result = brand_to_legal_name("ISP Desconocido")
        assert result == "ISP Desconocido"  # Fallback

    def test_claro_legal_name(self):
        assert brand_to_legal_name("Claro") == "CONECEL S.A."
