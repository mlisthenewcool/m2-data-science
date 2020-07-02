# Minimisation du risque empirique régularisé

On s'intéresse aux algorithmes d'apprentissage basés sur la minimisation du
risque empirique régularisé
$$\min_f \sum_i \ell(y_i ,f(x_i)) + \alpha \Omega(f) $$

## Régression ridge
* $\ell(y_i,f(x_i)) = (y_i - \beta^T x_i)^2$
* $\Omega(f) = \|\beta\|_2^2$ (norme $L_2$)

## Régression Lasso

* $\ell(y_i,f(x_i)) = (y_i - \beta^T x_i)^2$
* $\Omega(f) = \|\beta\|_1$ (norme $L_1$)

## Régression ElasticNet

* combine une régularization $L_1$ et une régularization $L_2$
* coefficients de régularisation $\alpha$ $(L_2)$ et $\rho$ $(L_1)$
* $\ell(y_i,f(x_i)) = (y_i - \beta^T x_i)^2$
* $\Omega(f) = \alpha \rho \|\beta\|_1 + \frac{\alpha(1-\rho)}{2}\|\beta\|_2$

## Sélection du paramètre $\alpha$

* Le choix de $\alpha$ est très important.
* On choisit $\alpha$ qui maximise l'erreur de cross validation parmi
plusieurs valeurs définies dans une grille.

TODO
* Appliquer et comparer la régression Ridge et la régression Lasso sur
le jeu de données boston.

* Afficher l'erreur de prédiction sur les données de test et les données
d'apprentissage en fonction du paramètre de régularisation.

* Afficher l'erreur de prédiction sur les données de test et les données
d'apprentissage en fonction du nombre de données d'apprentissage dans
les deux cas : $\alpha =0$ et $\alpha$ optimal choisi par cross-validation.

* Lire les sections 3.1 et 3.2 du chapitre 3 du livre "Pattern Recognition
and Machine Learning" (Bishop, 2006).
"""

## Résultats :
- Ridge : très grande importance à superplastic, cement, age, slag, ash
          très grande importance négative à water
- Lasso : très grande importance équivalente à cement, age, slag, ash
          très grande importance négative à water
- Eleastic : importance uniquement à cement (énorme), age, slag
