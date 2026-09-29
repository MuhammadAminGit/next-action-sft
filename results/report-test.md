# Results on ABCD test

3608 examples. Floor: always predicting `pull-up-account` gets 19.7% action accuracy. 84.5% of examples have every gold value present in the conversation. Brackets are 95% cluster-bootstrap intervals over conversations.

| System | Joint | Action | Values | Valid JSON |
|---|---|---|---|---|
| base-0shot-compact | 5.8 (5.0–6.6) | 15.7 (14.7–16.9) | 16.5 (15.2–17.8) | 69.3 (67.5–71.0) |
| base-0shot | 9.4 (8.5–10.4) | 18.1 (16.9–19.3) | 22.8 (21.5–24.4) | 76.6 (75.0–78.1) |
| base-5shot | 24.6 (23.1–26.0) | 34.9 (33.3–36.6) | 38.2 (36.6–39.9) | 98.9 (98.6–99.3) |
| lora | 71.3 (69.2–73.2) | 81.3 (79.2–83.2) | 75.5 (73.9–77.2) | 99.8 (99.6–99.9) |

## Improvement from fine-tuning (paired)

| Comparison | Metric | Change (points) | 95% CI | Excludes zero |
|---|---|---|---|---|
| lora vs base-0shot-compact | joint | +65.5 | +63.5 to +67.5 | yes |
| lora vs base-0shot-compact | action | +65.5 | +63.3 to +67.6 | yes |
| lora vs base-0shot-compact | values | +59.0 | +57.1 to +60.8 | yes |
| lora vs base-0shot | joint | +61.9 | +59.8 to +63.9 | yes |
| lora vs base-0shot | action | +63.2 | +60.9 to +65.5 | yes |
| lora vs base-0shot | values | +52.6 | +50.7 to +54.5 | yes |
| lora vs base-5shot | joint | +46.7 | +44.6 to +48.7 | yes |
| lora vs base-5shot | action | +46.4 | +44.2 to +48.5 | yes |
| lora vs base-5shot | values | +37.3 | +35.5 to +39.0 | yes |

## Joint accuracy by action

| Action | n | base-0shot-compact | base-0shot | base-5shot | lora |
|---|---|---|---|---|---|
| pull-up-account | 709 | 3.1 | 6.1 | 30.2 | 87.3 |
| verify-identity | 373 | 1.1 | 7.5 | 8.0 | 61.4 |
| search-faq | 242 | 0.4 | 0.4 | 0.4 | 85.1 |
| validate-purchase | 233 | 16.3 | 38.2 | 62.7 | 72.1 |
| enter-details | 211 | 0.0 | 0.0 | 2.4 | 70.6 |
| record-reason | 172 | 0.0 | 0.0 | 0.0 | 53.5 |
| ask-the-oracle | 171 | 0.0 | 0.0 | 0.0 | 88.9 |
| select-faq | 171 | 0.0 | 0.0 | 0.0 | 39.2 |
| update-order | 142 | 0.0 | 0.0 | 3.5 | 44.4 |
| membership | 136 | 43.4 | 23.5 | 73.5 | 86.8 |
| log-out-in | 92 | 47.8 | 63.0 | 76.1 | 68.5 |
| update-account | 92 | 1.1 | 2.2 | 18.5 | 41.3 |
| shipping-status | 86 | 29.1 | 10.5 | 66.3 | 51.2 |
| notify-team | 75 | 0.0 | 18.7 | 29.3 | 77.3 |
| offer-refund | 73 | 1.4 | 0.0 | 41.1 | 56.2 |
| try-again | 60 | 0.0 | 0.0 | 0.0 | 53.3 |
| make-purchase | 59 | 1.7 | 18.6 | 47.5 | 59.3 |
| send-link | 57 | 0.0 | 24.6 | 49.1 | 86.0 |
| promo-code | 53 | 0.0 | 0.0 | 17.0 | 92.5 |
| subscription-status | 53 | 0.0 | 1.9 | 5.7 | 58.5 |
| instructions | 49 | 2.0 | 0.0 | 0.0 | 69.4 |
| search-policy | 44 | 6.8 | 22.7 | 29.5 | 50.0 |
| make-password | 42 | 0.0 | 0.0 | 2.4 | 90.5 |
| search-pricing | 36 | 0.0 | 2.8 | 44.4 | 61.1 |
| search-membership | 35 | 20.0 | 37.1 | 60.0 | 71.4 |
| search-shirt | 34 | 0.0 | 0.0 | 0.0 | 97.1 |
| search-timing | 30 | 3.3 | 23.3 | 66.7 | 80.0 |
| search-jacket | 27 | 0.0 | 18.5 | 88.9 | 88.9 |
| search-jeans | 26 | 0.0 | 0.0 | 19.2 | 96.2 |
| search-boots | 25 | 4.0 | 4.0 | 88.0 | 88.0 |

## Most common action errors: lora

| Gold action | Predicted instead | Count |
|---|---|---|
| pull-up-account | search-faq | 28 |
| verify-identity | pull-up-account | 25 |
| validate-purchase | ask-the-oracle | 17 |
| update-order | enter-details | 17 |
| update-order | update-account | 16 |
| verify-identity | ask-the-oracle | 14 |
| try-again | enter-details | 12 |
| record-reason | ask-the-oracle | 12 |
| update-account | ask-the-oracle | 12 |
| shipping-status | ask-the-oracle | 12 |
