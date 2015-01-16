# dnnet
dnnet is a Java library for distributed backpropagation neural network on Apache Spark.

## Input Structure

```NeuralNetwork``` interface requires the training data as an instance of the ```LabelledData``` class. A helper class, ```IO```, can be used to create a ```LabelledData``` object from any file read through ```SparkContext```. Training data requires two files, one for input and one for output, such that observations from the two files are linked through a common serial key.

For example, let us consider the training files for XOR, located at ``` hdfs://localhost:9000/datasets/ ``` in the following format (described earlier)

**input.csv**
```csv
1,1.0,0.0
2,1.0,1.0
3,0.0,1.0
...
```
**output.csv**
```csv
1,1.0
2,0.0
3,1.0
...
```

```java
// SparkContext sc = ...
// IO.fetchLabelledData() automatically prompts to input the filenames
JavaRDD<LabelledDataPoint> labelledData = IO.fetchLabelledData(sc);
LabelledData trainSet = new LabelledData(labelledData);
```

For prediction, an instance of ```UnlabelledData``` is required, which can be created using a similar input file.

```java
JavaRDD<DoubleMatrix> unlabelledData = (IO.fetchUnlabelledData(sc));
UnlabelledData testSet = new UnlabelledData(unlabelledData);
```

## Usage
A detailed workflow is explained in the ```ml.dnnet.client.Main``` class.
```java
// Topography is a List<Integer> object specifying the neural network structure
NeuralNetwork nnet = new BackpropagationNeuralNetwork(topography, NeuralNetwork.Mode.REGRESSION);

// Gradient Checking
nnet.gradientCheck(trainSet);

// Training
// iterations = -1 if convergence is not reached within the specified number of epochs
int iterations = nnet.train(trainSet);

// Prediction
LabelledData predicted = nnet.predict(testSet);
List<LabelledDataPoint> result = predicted.getData().collect();
for (LabelledDataPoint line : result)
{
  System.out.println("Input : " + line.getX());
  System.out.println("Output : " + line.getY());
  System.out.println();
}
```

## External Libraries
* **_jblas_** for matrix/vector related computations

## Next Milestone
Export and import functionality for the ```NeuralNetwork``` instance (allowing continued usage of the trained model)

## Contributors
**Aditya Prasad**
