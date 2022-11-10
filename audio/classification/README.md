## Results

### audio features with GridSearchCV + oversampling + low variance
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
      <td>C=100 | penalty=l1 | solver=liblinear</td>
      <td>0.849352</td>
      <td>0.924135</td>
      <td>0.171667</td>
      <td>0.392042</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>criterion=entropy | max_depth=8 | max_features=auto | n_estimators=10</td>
        <td>0.844521</td>
        <td>0.862922</td>
        <td>0.151667</td>
        <td>0.358932</td>
      </tr>
  <tr>
      <td>SVM Linear</td>
      <td>C=1 | gamma=0.001 | kernel=rbf</td>
      <td>0.935441</td>
      <td>0.885148</td>
      <td>0.060000</td>
      <td>0.193415</td>
    </tr>
  
</table>

### audio features with GridSearchCV + oversampling + low variance + f1_regression
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
      <td>C=10 | penalty=l1 | solver=liblinear</td>
      <td>0.848694</td>
      <td>0.968143</td>
      <td>0.17</td>
      <td>0.383202</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>criterion=entropy | max_depth=8 | max_features=auto | n_estimators=10</td>
        <td>0.859793</td>
        <td>0.908554</td>
        <td>0.146667</td>
        <td>0.356800</td>
      </tr>
  <tr>
      <td>SVM Linear</td>
      <td>C=1 | gamma=0.001 | kernel=rbf</td>
      <td>0.930065</td>
      <td>0.877079</td>
      <td>0.063333</td>
      <td>0.192697</td>
    </tr>
  
</table>