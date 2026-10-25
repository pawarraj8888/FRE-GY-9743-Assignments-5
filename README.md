# HW5: Fixed-Income Valuation and First-Order Risk

Complete the five TODOs in the following files.

| TODO | File | Method |
|---|---|---|
| 1 | [yield_curve_model.py](fixedincomelib/yield_curve/yield_curve_model.py) | `YieldCurveModelComponent.discount_factor_gradient_wrt_state` |
| 2 | [yield_curve_model.py](fixedincomelib/yield_curve/yield_curve_model.py) | `YieldCurve.discount_factor_gradient_wrt_state` |
| 3 | [valuation_engine.py](fixedincomelib/yield_curve/valuation_engine.py) | `ValuationEngineProductFixedAccrued.__init__` |
| 4 | [valuation_engine.py](fixedincomelib/yield_curve/valuation_engine.py) | `ValuationEngineProductFixedAccrued.calculate_value` |
| 5 | [valuation_engine.py](fixedincomelib/yield_curve/valuation_engine.py) | `ValuationEngineProductFixedAccrued.calculate_risk` |

## Assignment 5 submission: Raj Pawar (`Raj_Pawar_Branch`)

| Deliverable | Location |
|---|---|
| TODO 1 and 2: the component's local discount-factor gradient `dD/df_i = -D * (length of node i's bucket inside [0, T])`, and the model's gradient through the reference chain by the product rule, written into component-ordered blocks with the `scaler` / `accumulate` semantics | `fixedincomelib/yield_curve/yield_curve_model.py` |
| TODO 3 to 5: the fixed-accrued valuation engine (funding curve from the `FUNDING INDEX PARAMETER`, `PV = C D(T_p)` with the same-day and past cases, `risk = C dD(T_p)/dstate`) | `fixedincomelib/yield_curve/valuation_engine.py` |
| Professor's notebook executed top to bottom with the five TODOs in place (no notebook cell needed changing) | `hw5_valuation_and_first_order_risk.ipynb` |
| Tests: gradients against a by-hand bucket-length formula and central bump-and-reval, the product rule, overwrite and accumulate, block allocation, validation errors; engine PV, cash and risk for future, same-day and past payments, the sign of the notional, the reports and the registry dispatch | `tests/test_yield_curve_gradient.py`, `tests/test_valuation_engine.py` |
| Write-up: Part I (implicit-function differentiation and the vol-interpolator calibration) and a summary of Part II | `writeup/Assignment5_Writeup_RajPawar.pdf` (source `.tex` alongside) |
| Project page, numbers generated from the library | `docs/index.html` (built by `docs/build_page.py`) |

### How to run

```bash
pip install -r requirements.txt                         # Python 3.11, QuantLib 1.41
jupyter notebook hw5_valuation_and_first_order_risk.ipynb   # Run All
python tests/test_yield_curve_gradient.py               # or, with pytest installed: python -m pytest tests -q
python tests/test_valuation_engine.py
```

Only the two `TODO` files changed. Besides the five `TODO` bodies I added a class docstring and two small accessors (`funding_index`, `cash_amount`) that the tests use; the model, builder, data, product, report and API layers are unchanged.
