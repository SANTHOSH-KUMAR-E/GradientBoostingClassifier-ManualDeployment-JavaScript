// Refer GBClassifierModelStructure.js file for GB Tree structure (gb_structure), sample X_train records (test10) and Learning_Rate

// Server side Script 
// Prepare 'ValueArray' by getting user data from Client side and call this function.
function Predict(ValueArray) {
  // Converting time (HH:MM) to minutes (0 to 14400)
  ValueArray[6] = (ValueArray[6].split(':')[0] * 60) + (ValueArray[6].split(':')[1] * 1);
  ValueArray[7] = (ValueArray[7].split(':')[0] * 60) + (ValueArray[7].split(':')[1] * 1);

  // MinMax Scalar
  ValueArray[1] = ((ValueArray[1] * 1) - 1) / (31 - 1);
  ValueArray[2] = ((ValueArray[2] * 1) - 1) / (7 - 1);

  // Frequency Encoded (Done in front end, by setting the frequency as value for corresponding dropdown labels)
  ValueArray[3] = (ValueArray[3] * 1);
  ValueArray[0] = (ValueArray[0] * 1);
  ValueArray[11] = (ValueArray[11] * 1);
  ValueArray[15] = (ValueArray[15] * 1);

  // Feature Transformation
  ValueArray[4] = Math.cbrt(ValueArray[4] * 1);
  ValueArray[5] = Math.log(ValueArray[5] * 1);
  ValueArray[16] = Math.pow((ValueArray[16] * 1), 3);

  // Standard Scalar
  // ScaleArray - Shape : ( [ mean, standard deviation ], n_features ) --> ( 2, 18 )
  // Mean = 0, Std = 1, for those features which doesn't require scaling
  
  ScaleArray = [[ 0, 0, 0, 0, -4.48541610e-01,  5.26745748e+00,  
                    8.30488227e+02, 9.10254381e+02,  4.15495108e+01,  3.06773465e+01, 5.81878651e+01, 0, 
                    1.20508547e+01, 0, 3.00918310e+01, 0, 2.38756486e+05,  4.99314984e+00],
                [ 1, 1, 1, 1, 1.98131375e+00,  5.65632200e-01,  
                    2.99871664e+02, 3.45860197e+02,  7.76191155e+00,  1.19208650e+01, 2.33795502e+01, 1, 
                    5.93881074e+00, 1, 2.91169450e-01, 1, 1.18471648e+05,  1.42228659e+00]];

  for (var x=0; x<ValueArray.length; x++) {
    ValueArray[x] = ((ValueArray[x] * 1) - ScaleArray[0][x]) / (ScaleArray[1][x]);
  }

  // Prediction using Gradient Boosting - Binomial Classification
  var Pred_Class = GB_Single_Row_Predict(ValueArray);

  if (Pred_Class == 0) {
    return "The Taxi-Out Run time of this flight is LESS than 20 mins (Less Delay)";
  } else {
    return "The Taxi-Out Run time of this flight is GREATER than 20 mins (High Delay)";
  }
}


// Predict Function for Gradient Boosting Classifier - Binomial Classification
function GB_Single_Row_Predict(x) {
  
    var pred_value = - 0.01343986;  // Initial Prediction (log odds probability)
    var n = 250;                    // No. of estimators

    // Y = Init_prediction + (learning_rate * residual prediction of estimator 1) + (learning_rate * residual prediction of estimator 2) + ....
    for (var est=0; est<n; est++) {
      pred_value = pred_value + (learning_rate * GB_Single_Tree_Predict(x, est));
    }
  
    // Converting raw prediction to probability
    var proba = Math.exp(pred_value) / ( 1 + Math.exp(pred_value) );
  
    if (proba < 0.5) {
      return 0;
    } else {
      return 1;
    }
}


// Residual Prediction for each estimator
function GB_Single_Tree_Predict(x, est_id) {
  
  // Starting from root node
  var current_node = 0;

  while (1==1) {
    
    // if current node is Leaf node (left child = right child), return the residual value
    // else, move to next decision node using Feature and Threshold value of current node.
    if ( gb_structure[est_id][current_node][0] == gb_structure[est_id][current_node][1] ) {
      return gb_structure[est_id][current_node][4];
    } else {
      var feature = x[gb_structure[est_id][current_node][2]];
      var thres = gb_structure[est_id][current_node][3];
      if ( feature <= thres ) {
        current_node = gb_structure[est_id][current_node][0];
      } else {
        current_node = gb_structure[est_id][current_node][1];
      }
    }
  }
}


// Check the built function prediction against the python model prediction
function deploy_check() {
  Logger.log(GB_Single_Row_Predict(test10[0]));
}
