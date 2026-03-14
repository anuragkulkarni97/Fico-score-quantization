FICO Score Quantization: 

Optimal Bucket Construction

A comprehensive credit risk project to find the optimal FICO score boundaries that maximally separate borrower default risk into discrete rating tiers. This repository implements and compares two bucketing algorithms, MSE Minimization and Log-Likelihood Dynamic Programming, and identifies the rating system that best serves real-world loan pricing and capital allocation decisions.

The model is applied to a real loan dataset from the JPMorgan Chase Quantitative Research, producing a clean 5-tier rating system where default probability ranges from 6.4% (best credit) to 87.9% (highest risk), with near-perfect monotonic separation between every tier.

Key Features: 

MSE Minimization Bucketing: Groups FICO scores into buckets by minimizing the within-bucket variance of default rates, ensuring that borrowers assigned the same rating tier have similar risk profiles.

Log-Likelihood Dynamic Programming (Winner): Solves the bucketing problem as a global optimization using dynamic programming. Maximizes the statistical likelihood of observing the actual default pattern given the bucket assignments, producing cleaner separation than MSE at every tier boundary.

Boundary Comparison: Both methods are visualized side-by-side against the rolling default rate curve, showing exactly where LL boundaries outperform MSE boundaries in capturing real default risk transitions.

Optimal Bucket Selection: An elbow curve plotting log-likelihood and MSE against number of buckets (2 through 10) identifies 5 as the optimal bucket count, the point of maximum improvement before diminishing returns set in.

Rating Grade Assignment: Each borrower is assigned a rating from R1 (best credit, FICO 721–851, PD = 6.4%) through R5 (highest risk, FICO 484–603, PD = 87.9%), with population counts and default sub-counts shown for every tier.

5-Panel Visualization: A publication-quality chart covering FICO distribution by default status, PD per rating bucket, population per bucket, boundary comparison, and the objective score elbow curve, all generated from a single Python script.

Results:

5-Bucket Log-Likelihood DP Solution
RatingFICO RangeProbability of DefaultPopulationR1 — Best Credit721 – 8516.4%1, 255R2694 - 72019.9%858R3651 – 69334.5%1, 350R4604 - 65061.5%1, 050R5 - Highest Risk484 – 60387.9%487

Tech Stack: 

Python 3.x


NumPy: For dynamic programming table construction and array operations.

SciPy: For statistical helper functions used in the log-likelihood computation.

Matplotlib: For generating the 5-panel visualization output.

Pandas: For loading and preprocessing the loan dataset.

How to Run: 

Clone this repository to your local machine.

Install the required libraries: pip install numpy scipy matplotlib pandas

Run the script: python fico_quantization.py

The script will print the rating tier boundaries and default rates to the terminal, and save the 5-panel chart as a PNG file.


Future Work: 
This project provides a strong foundation for production credit rating systems. 

Future improvements could include:

Incorporating More Features: Extending the bucketing beyond FICO alone to jointly optimize boundaries across FICO score, debt-to-income ratio, and loan amount, building a multi-dimensional rating grid.

Bayesian Smoothing: Applying Laplace smoothing to default rate estimates within small buckets (like R2 with n=858) to reduce variance in probability of default estimates.

Regulatory Alignment: Mapping the 5 output tiers to Basel III internal rating grades and computing the minimum required capital per tier using the IRB approach.

Integration with Loan Default Model: Using the PD outputs from this quantization as direct inputs into an expected loss calculator, closing the loop between score bucketing and portfolio-level credit risk measurement.

About

Built by Anurag Kulkarni as part of the JPMorgan Chase Quantitative Research.

Connect on LinkedIn: https://www.linkedin.com/in/anurag-kulkarni97/

GitHub: AnalyticalAnurag97
