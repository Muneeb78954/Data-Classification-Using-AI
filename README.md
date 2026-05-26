What's inside
Step 1 — Load Dataset
A synthetic 150-sample Iris-like dataset with 4 features (sepalLength, sepalWidth, petalLength, petalWidth) and 3 balanced classes. You can preview the first rows and class distribution.
Step 2 — Train/Test Split
A slider lets you control the test ratio (10–40%). Stats update in real time showing exactly how many samples go to training vs. testing — with equivalent sklearn code shown.
Step 3 — Train the Model (KNN)
K-Nearest Neighbours is implemented from scratch. Adjust k (number of neighbours) with a slider and see how it affects results. The algorithm explains its own logic inline.
Step 4 — Evaluate

Accuracy score with colour-coded feedback
Confusion matrix showing per-class performance
Visual prediction strip — green ✓ / red ✗ for each test sample


Key learning takeaways
Concept               Meaning
Features (X)          The 4 numerical inputs the model uses
Labels (y)            The flower class to predict
Train/Test split      Prevents "cheating" by evaluating on unseen data
KNN                   No training phase — classifies by nearest neighbour vote
Accuracy              % of test samples correctly classified

Try setting k=1 vs k=15 and a small vs large test ratio to see how they affect accuracy!
