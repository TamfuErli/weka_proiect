package Proiektua;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class ProbaFinala {
    public  static void main(String[] args){
        try{
            String modelPath = args[0];
            String test_blind_csv = args[1];
            String predikzioak = args[2];
            String hiztegi = args[3];

            DataSource src = new DataSource(test_blind_csv);
            Instances testRaw = src.getDataSet();
            testRaw.setClassIndex(2);

            Instances test_bow_fss = FssDevTest.applyFssWithDictionary(testRaw, hiztegi);
            test_bow_fss.setClassIndex(2);

            BayesNet modeloa = (BayesNet) SerializationHelper.read(modelPath);
            Baseline.predikzioakEgin(predikzioak, modeloa, test_bow_fss);

        } catch (Exception e) {
            System.out.println("ERROREA");
            e.printStackTrace();
        }
    }
}
