"""Built-in decision ("System One") provider plugin: TypeSafe, Kev, Laya.

All three speak the same typed-decision contract — a ``state`` plus a map of
``noul`` / ``choice`` / ``score`` questions in, typed answers with calibrated
probabilities out.  They differ only in where the weights run:

* ``typesafe`` — hosted Jev, billed per input token.
* ``kev`` — self-hosted, byte-identical ``/v1/systemone`` endpoint.
* ``laya`` — in-process open weights, no server at all.
"""

from __future__ import annotations

from ...drivers.provider_descriptors import DriverSpec, ProviderDescriptor
from ..base import ProviderPlugin


class DecisionPlugin(ProviderPlugin):
    name = "decision_builtin"
    version = "1.0.0"

    def descriptors(self) -> list[ProviderDescriptor]:
        descs: list[ProviderDescriptor] = []

        # TypeSafe — hosted Jev.
        typesafe_kw = {"api_key": "typesafe_api_key", "base_url": "typesafe_base_url"}
        descs.append(
            ProviderDescriptor(
                name="typesafe",
                decision_sync=DriverSpec(
                    "typesafe_decision_driver.TypeSafeDecisionDriver",
                    typesafe_kw,
                    "typesafe_decision_model",
                ),
                decision_async=DriverSpec(
                    "async_typesafe_decision_driver.AsyncTypeSafeDecisionDriver",
                    typesafe_kw,
                    "typesafe_decision_model",
                ),
                display_name="TypeSafe (Jev)",
                is_configured_check="typesafe_api_key",
                models_dev_name=None,
            )
        )

        # Kev — self-hosted Jev-compatible server.
        kev_kw = {"api_key": "kev_api_key", "base_url": "kev_base_url"}
        descs.append(
            ProviderDescriptor(
                name="kev",
                decision_sync=DriverSpec(
                    "kev_decision_driver.KevDecisionDriver",
                    kev_kw,
                    "kev_decision_model",
                ),
                decision_async=DriverSpec(
                    "async_kev_decision_driver.AsyncKevDecisionDriver",
                    kev_kw,
                    "kev_decision_model",
                ),
                display_name="Kev (self-hosted)",
                always_available=True,
                models_dev_name=None,
            )
        )

        # Laya — in-process open weights.
        laya_kw = {"device": "laya_device"}
        descs.append(
            ProviderDescriptor(
                name="laya",
                decision_sync=DriverSpec(
                    "laya_decision_driver.LayaDecisionDriver",
                    laya_kw,
                    "laya_decision_model",
                ),
                decision_async=DriverSpec(
                    "async_laya_decision_driver.AsyncLayaDecisionDriver",
                    laya_kw,
                    "laya_decision_model",
                ),
                display_name="Laya (local)",
                always_available=True,
                models_dev_name=None,
            )
        )

        return descs
