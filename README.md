# Online GentleBoost

This repository contains code used for the Online GentleBoost technical report. 

Gentleboost was first conceived in "Additive logistic regression: a statistical view of boosting",  Friedman, Jerome; Hastie, Trevor; Tibshirani, Robert (2000). The approach is the form of AdaBoost except where Newton Steps are used instead of exact optimisation. 

In our technical report we explore a theoretical sound and straightforward way to convert the GentleBoost algorithm to the online setting, which is outlined below:

+-------------------------------------------------------------------------------------------------------------+
| **Online GentleBoost**                                                                                      |
+=============================================================================================================+
| 1. Start $F(x) =0 $, with hyperparamter $\alpha \in (0, \exp(1))$                                           |
| 2. For incoming instance $x_i, y_i$, reset weight $w_i = 1$:                                                | 
| 3. Repeat for $m = 1,2, \dots, M$:                                                                          |
|    a. Fit the regression function $f_m(x)$ by weighted least-squares of $y_i$ to $x_i$ with weights $w_i$.  |
|    b. Update $F(x) \leftarrow F(x) + f_m(x)$.                                                               |
|    c. Update $w_i \leftarrow \hat{\alpha} w_i$ and renormalize                                       |
| 4. Go back to 2. if there are additional instances                                                          |
| 5. Finally output the classifier $\text{sign}(F(x)) = \text{sign}(\sum_{m=1}^M f_m(x))$.                    |
+-------------------------------------------------------------------------------------------------------------+

Empirically GentleBoost does indeed create a stronger learner from the weak learner. Compared to other boosting approaches, GentleBoost may be a bit _too_ gentle, and from multiple experiments appears to be inferior to other approaches. Online Gentleboost is extremely straightforward to implement and understand and may have implications in other contexts such as Deep Learning architectures where we boost models which used parameter sharing. That can be a discussion and topic for exploration at a later stage. 
