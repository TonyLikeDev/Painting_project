# Gradient of the Sinkhorn cost relative to the pixel loss

120 probes. Each cell is the median over probes of ||grad Sinkhorn|| / ||grad pixel|| (unweighted; multiply by the weight beta).

| Moment | all strokes | newest stroke | newest stroke, position |
| :--- | ---: | ---: | ---: |
| first step of a new stroke | 0.00718 | 0.0034 | 0.00316 |
| last step of a stroke | 0.0087 | 0.00429 | 0.00423 |
