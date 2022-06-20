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
