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

### bag-of-words with GridSearchCV
<table>
    <tr>
      <td><b>Model</b></td>
      <td><b>Tuned hyperparameters</b></td>
      <td><b>F1 Score</b></td>
      <td><b>Recall</b></td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>C=1 and penalty=l2</td>
      <td>0.4325</td>
      <td>0.4244</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>N_estimators=1</td>
        <td>0.2634</td>
        <td>0.2594</td>
      </tr>
  <tr>
      <td>Decision Tree</td>
      <td>Criterion=entropy and max_depth=6</td>
      <td>0.4450</td>
      <td>0.4763</td>
    </tr>
  <tr>
      <td>SVM Linear</td>
      <td>C=1 and kernel=linear</td>
      <td>0.3946</td>
      <td>0.3928</td>
    </tr>
  
</table>

### tf-idf with GridSearchCV
<table>
    <tr>
      <td><b>Model</b></td>
      <td><b>Tuned hyperparameters</b></td>
      <td><b>F1 Score</b></td>
      <td><b>Recall</b></td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>C=1000 and penalty=l1</td>
      <td>0.3812</td>
      <td>0.3094</td>
    </tr>
    <tr>
        <td>Random Forest</td>
        <td>N_estimators=1</td>
        <td>0.2633</td>
        <td>0.2490</td>
      </tr>
  <tr>
      <td>Decision Tree</td>
      <td>---</td>
      <td>---</td>
      <td>---</td>
    </tr>
  <tr>
      <td>SVM Linear</td>
      <td>---</td>
      <td>---</td>
      <td>---</td>
    </tr>
  
</table>
