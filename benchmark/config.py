from river import ensemble, evaluate, forest, linear_model, naive_bayes, optim, preprocessing, tree

from gentleboost.gentleboost import GentleBoostClassifier

N_CHECKPOINTS = 50

LEARNING_RATE = 0.005

TRACKS = [
    evaluate.BinaryClassificationTrack(),
]

MODELS = {
    "Binary classification": {
        "Logistic regression": (
            preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.SGD(LEARNING_RATE))
        ),
        "Aggregated Mondrian Forest": forest.AMFClassifier(seed=42),
        "ALMA": preprocessing.StandardScaler() | linear_model.ALMAClassifier(),
        "Naive Bayes": naive_bayes.GaussianNB(),
        "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
        "Hoeffding Adaptive Tree": tree.HoeffdingAdaptiveTreeClassifier(seed=42),
        "Adaptive Random Forest": forest.ARFClassifier(seed=42),
        "ADWIN Bagging": ensemble.ADWINBaggingClassifier(tree.HoeffdingTreeClassifier(), seed=42),
        "AdaBoost": ensemble.AdaBoostClassifier(tree.HoeffdingTreeClassifier(), seed=42),
        "GentleBoost": GentleBoostClassifier(tree.HoeffdingTreeClassifier(), seed=42),
        "Bagging": ensemble.BaggingClassifier(tree.HoeffdingAdaptiveTreeClassifier(bootstrap_sampling=False), seed=42),
        "Leveraging Bagging": ensemble.LeveragingBaggingClassifier(tree.HoeffdingTreeClassifier(), seed=42),
    },
}
