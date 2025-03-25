package Proiektua;

import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.*;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.core.Instances;
import java.util.Random;

public class InferentziaOptimzatu {
    /**
     * Klase maioritarioaren indizea eskurantzen du
     */
    public static int klaseMayoritarioa(Instances data) {
        int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
        int maxIndex = 0;
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] > counts[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }

    /**
     * Klase minoritarioaren indizea eskurantzen du
     */
    public static int klaseMinoritarioa(Instances data) {
        int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
        int minIndex = 0;
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] < counts[minIndex] && counts[i] > 0) minIndex = i;
        }
        return minIndex;
    }

    private static CostMatrix crearMatrizCostos(int numClasses, int claseMinoritaria) {
        CostMatrix matrix = new CostMatrix(numClasses);
        matrix.initialize();
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (i != j && i == claseMinoritaria) {
                    matrix.setCell(i, j, 2.0); // Honekin clase minoritarioaren errorea x2 kostuarekin penalizatzen  da
                } else {
                    matrix.setCell(i, j, 1.0); // Error estándar
                }
            }
        }
        return matrix;
    }

    public static Classifier parametroakOptimizatu(Instances train, Instances dev) throws Exception{

        SMO modeloHoberena = new SMO();
        Kernel bestKernel = null;
        double bestFMeassure = -1;

        int minClassIndex = klaseMinoritarioa(dev);
        CostMatrix costMatrix = crearMatrizCostos(train.numClasses(), minClassIndex);

        Kernel[] kernelDesberdinak = {
                new PolyKernel(),
                new RBFKernel(),
                new Puk()
        };

        for (Kernel kernel : kernelDesberdinak){
            SMO model = new SMO();
            model.setKernel(kernel);
            CostSensitiveClassifier costModel = new CostSensitiveClassifier();
            costModel.setClassifier(model);
            costModel.setCostMatrix(costMatrix);
            costModel.buildClassifier(train);

            

            if()
        }
    }
}
