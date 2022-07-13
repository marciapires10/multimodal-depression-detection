## Results

### bag-of-words
<table>
    <tr>
      <td><b>Model</b></td>
      <td><b>F1 Score</b></td>
      <td><b>Recall</b></td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>0.4559</td>
      <td>0.4492</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>0.2634</td>
        <td>0.2594</td>
      </tr>
  <tr>
      <td>Decision Tree</td>
      <td>0.3664</td>
      <td>0.3619</td>
    </tr>
  <tr>
      <td>SVM Linear</td>
      <td>0.4274</td>
      <td>0.4306</td>
    </tr>
  
</table>

### bag-of-words with GridSearchCV + undersampling
<table>
    <tr>
      <td><b>Model</b></td>
      <td><b>Tuned hyperparameters</b></td>
      <td><b>F1 Score</b></td>
      <td><b>Recall</b></td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>C=0.01 and penalty=l2</td>
      <td>0.6969</td>
      <td>0.7356</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>N_estimators=10</td>
        <td>0.5338</td>
        <td>0.5511</td>
      </tr>
  <tr>
      <td>Decision Tree</td>
      <td>Criterion=entropy and max_depth=2</td>
      <td>0.6531</td>
      <td>0.6950</td>
    </tr>
  <tr>
      <td>SVM Linear</td>
      <td>C=1 and kernel=linear</td>
      <td>0.6453</td>
      <td>0.6956</td>
    </tr>
  
</table>


### bag-of-words with GridSearchCV + oversampling
<table>
    <tr>
      <td><b>Model</b></td>
      <td><b>Tuned hyperparameters</b></td>
      <td><b>F1 Score</b></td>
      <td><b>Recall</b></td>
      <td><b>MAE</b></td>
      <td><b>RMSE</b></td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>C=1000 | penalty=l2 | solver=lbfgs</td>
      <td>0.880211</td>
      <td>0.913542</td>
      <td>0.120000</td>
      <td>0.330855</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>criterion=entropy | max_depth=8 | max_features=auto | n_estimators=10</td>
        <td>0.808555</td>
        <td>0.841766</td>
        <td>0.189910</td>
        <td>0.407796</td>
      </tr>
  <tr>
      <td>Decision Tree</td>
      <td>criterion=gini and max_depth=8</td>
      <td>0.755394</td>
      <td>0.805180</td>
      <td>0.248333</td>
      <td>0.490598</td>
    </tr>
  <tr>
      <td>SVM Linear</td>
      <td>C=10 | gamma=0.001 | kernel=rbf</td>
      <td>0.921858</td>
      <td>0.893289</td>
      <td>0.071667</td>
      <td>0.221367</td>
    </tr>
  
</table>


### tf-idf with GridSearchCV + undersampling
<table>
    <tr>
      <td><b>Model</b></td>
      <td><b>Tuned hyperparameters</b></td>
      <td><b>F1 Score</b></td>
      <td><b>Recall</b></td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>C=100 and penalty=l1</td>
      <td>0.5108</td>
      <td>0.6744</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>N_estimators=1</td>
        <td>0.6138</td>
        <td>0.6489</td>
      </tr>
  <tr>
      <td>Decision Tree</td>
      <td>Criterion=entropy and max_depth=4</td>
      <td>0.6138</td>
      <td>0.6489</td>
    </tr>
  <tr>
      <td>SVM Linear</td>
      <td>C=10 and kernel=linear</td>
      <td>0.6712</td>
      <td>0.7011</td>
    </tr>
  
</table>


### tf-idf with GridSearchCV + oversampling
<table>
    <tr>
      <td><b>Model</b></td>
      <td><b>Tuned hyperparameters</b></td>
      <td><b>F1 Score</b></td>
      <td><b>Recall</b></td>
      <td><b>MAE</b></td>
      <td><b>RMSE</b></td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>C=1 | penalty=l2</td>
      <td>0.944773</td>
      <td>0.904241</td>
      <td>0.046667</td>
      <td>0.160702</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>criterion=gini | max_depth=8 | max_features=auto | n_estimators=10</td>
        <td>0.783086</td>
        <td>0.783902</td>
        <td>0.205000</td>
        <td>0.438345</td>
      </tr>
  <tr>
      <td>Decision Tree</td>
      <td>criterion=entropy | max_depth=8</td>
      <td>0.755538</td>
      <td>0.812136</td>
      <td>0.255</td>
      <td>0.497348</td>
    </tr>
  <tr>
      <td>SVM Linear</td>
      <td>C=1 | degree=3 | kernel=poly</td>
      <td>0.938433</td>
      <td>0.890117</td>
      <td>0.058333</td>
      <td>0.185615</td>
    </tr>
  
</table>