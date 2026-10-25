"""Analytic valuation of a signed fixed-accrual cashflow on an IFR model."""

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    """PV and raw first-order IFR risk of one fixed-accrual cashflow.

    The cashflow pays C = notional * accrued on its payment date T_p. Discounting uses the
    funding curve named by the FUNDING INDEX PARAMETER, so PV = C * D(T_p) and the raw risk
    is C * dD(T_p)/dstate, written block by block in the model's component order. A payment
    on the value date is cash (PV = cash = C, no risk); a payment in the past is worth nothing.
    """

    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        if not isinstance(product, ProductFixedAccruedCashflow):
            raise TypeError("ValuationEngineProductFixedAccrued needs a ProductFixedAccruedCashflow")
        if model.model_type != ModelType.YIELD_CURVE.to_string():
            raise TypeError("ValuationEngineProductFixedAccrued needs a yield-curve model")
        if not valuation_parameters_collection.has_vp_type(FundingIndexParameter._vp_type):
            raise ValueError("A FUNDING INDEX PARAMETER is needed to choose the discounting curve")
        funding_parameter = valuation_parameters_collection.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = funding_parameter.get_funding_index(product.currency)
        self.payment_date_ = product.payment_date
        # C = N * alpha; the sign comes from the notional, there is no separate fixed rate.
        self.cash_amount_ = product.notional * product.accrued

    @property
    def funding_index(self):
        return self.funding_index_

    @property
    def cash_amount(self) -> float:
        return self.cash_amount_

    def pays_in_the_future(self) -> bool:
        return self.payment_date_ > self.value_date

    def calculate_value(self) -> None:
        if self.payment_date_ < self.value_date:
            # Already paid: nothing is left on the contract.
            self.value_, self.cash_ = 0.0, 0.0
        elif not self.pays_in_the_future():
            # Pays today: a known cash amount, nothing to discount.
            self.value_, self.cash_ = self.cash_amount_, self.cash_amount_
        else:
            discount_factor = self.model.discount_factor(
                self.funding_index_, self.payment_date_, self.product_.currency.code()
            )
            self.value_, self.cash_ = self.cash_amount_ * discount_factor, 0.0

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        if self.pays_in_the_future():
            # PV = C * D(T_p): the raw risk is C times the model's DF gradient, block by block.
            self.model.discount_factor_gradient_wrt_state(
                self.funding_index_, self.payment_date_, gradient,
                scaler=scaler * self.cash_amount_, accumulate=accumulate,
            )
            return
        # Cash today or already paid: no discount factor, so no state risk. Overwriting still
        # has to leave correctly sized zero blocks behind; accumulating adds nothing.
        self.model.resize_gradient(gradient)
        if not accumulate:
            for block in gradient:
                block.fill(0.0)


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)
