# Used Car Pricing: Should the Dealer Buy This Car?

Linear regression predicting a used car's sale price, and a buy/pass rule for auction lots.

## How to run
pip install -r requirements.txt
python analysis.py

## Files
- analysis.py: loads data, 70/30 split, fits the model, evaluates it, applies the decision rule to 40 auction lots
- used_car_sales.csv: 240 cars the dealer sold last year (model data)
- auction_lots.csv: 40 auction lots (decision test only)
- requirements.txt: Python packages
- results.txt: output of the final run
- pred_vs_actual.png: test-set chart
- auction_decisions.csv: buy/pass decision and profit per lot

## Notes
Random seed is fixed (42), so results reproduce exactly.