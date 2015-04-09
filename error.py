from sklearn.metrics import mean_squared_error

class Error:
    def __init__(self, examples):
        self.RMSE = 0        # final kaggle score
        self.RMSE_time = 0  # RMSE on absolute value of position prediction   
        self.accuracy = 0    # percent correctness
        self.examples = examples

        # Estimate final Kaggle score
        predictions = [x.prediction for x in examples]
        observations = [x.observation for x in examples]
        self.RMSE = self._rootMeanSquaredError(predictions, observations)

        # Estimate RMSE on times only
        predictions = [x.predicted_time for x in examples]
        observations = [x.observed_time for x in examples]
        self.RMSE_time = self._rootMeanSquaredError(predictions, observations)

        # Success ratio of correctness
        predictions = [x.predicted_correctness for x in examples]
        observations = [x.observed_correctness for x in examples]
        nCorrect = 0
        for p,o in zip(predictions, observations):
            if p*o == 1:
                nCorrect = nCorrect + 1
        self.accuracy = float(nCorrect) / float(len(predictions) )

    def confusion(self):
        p = [x.predicted_correctness for x in self.examples]
        o = [x.observed_correctness for x in self.examples]

        trueNegatives =  str(len([x for x,y in zip(p,o) if x == y and x == -1]))
        falseNegatives = str(len([x for x,y in zip(p,o) if x != y and x == -1]))
        truePositives =  str(len([x for x,y in zip(p,o) if x == y and x == 1]))
        falsePositives = str(len([x for x,y in zip(p,o) if x != y and x == 1]))

        print "          0     1  "
        print "      0  "+trueNegatives+"   "+falseNegatives
        print "      1  "+falsePositives+"   "+truePositives

    def _rootMeanSquaredError(self, predictions, observations):
        return mean_squared_error(predictions, observations) ** 0.5


