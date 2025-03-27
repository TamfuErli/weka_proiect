package Proiektua;
import com.sun.xml.bind.v2.runtime.unmarshaller.IntArrayData;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.supervised.instance.Resample;

import javax.xml.crypto.Data;
import java.util.Random;


public class ereduarenKakitateEstimazioa {
    public void main(String argv[ ]){
        try{
            //Datuak kargatu
            DataSource source_train = new DataSource(argv[0]);
            Instances train_data = source_train.getDataSet();
            //mirar donde esta puesta la clase para editar esto
            train_data.setClassIndex(train_data.numAttributes() - 1);

            DataSource source_dev = new DataSource(argv[1]);
            Instances dev_data = source_dev.getDataSet();
            //mirar donde esta puesta la clase para editar esto
            dev_data.setClassIndex(dev_data.numAttributes() - 1);

            DataSource source_attr = new DataSource(argv[2]);
            Instances attr_data = source_attr.getDataSet();
            attr_data.setClassIndex(attr_data.numAttributes() - 1);

            //Datuak egokitu nahi dugun atributuentzako
            int[] attrTrain = lortuIndizeak(dev_data, attr_data);
            int[] attrDev = lortuIndizeak(dev_data, attr_data);

            Instances trainEgokituta = lortuDatuakEgokituta(train_data, attrTrain);
            Instances devEgokituta = lortuDatuakEgokituta(dev_data, attrDev);

            //Bi datu sortak bateratu
            Instances dataBerria = new Instances(trainEgokituta);
            for(int i=0; i<devEgokituta.numInstances(); i++){
                dataBerria.add(devEgokituta.instance(i));
            }

            //Modeloaren estimazioa egin
            holdOutRepeated(dataBerria);
            ezZintzoa(dataBerria);
            fCV(dataBerria);


        }catch (Exception e){}
    }

    private int[] lortuIndizeak(Instances datuak,Instances hiztegia) throws Exception{

        Instances datuakEgokituta = datuak;
        Instances datuKop=hiztegia;
        int[] atrInd = new int[datuKop.numAttributes()];

        for (int i = 0; i < datuKop.numAttributes(); i++) {
            String izena = datuKop.attribute(i).name();
            int index = datuKop.attribute(izena).index();
            atrInd[i] = index;
        }

        return atrInd;
    }

    private Instances lortuDatuakEgokituta (Instances datuak, int[] attrInd) throws Exception{

        Remove remove = new Remove();
        remove.setInputFormat(datuak);
        remove.setAttributeIndicesArray(attrInd);
        remove.setInvertSelection(true);
        Instances datuakEgokituta= Filter.useFilter(datuak,remove);

        return datuakEgokituta;
    }

    private void ezZintzoa(Instances datuak) throws Exception{

        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(datuak);

        Evaluation eval = new Evaluation(datuak);
        eval.evaluateModel(nb, datuak);

        Double fMeasure = eval.fMeasure(datuak.classIndex());
        Double precision = eval.precision(datuak.classIndex());
        Double recall = eval.recall(datuak.classIndex());
    }
    private void holdOutRepeated(Instances datuak) throws Exception{

        Double avgFmeasure = 0.0;
        Double avgPrecission = 0.0;
        Double avgRecall = 0.0;

        for (int i=0; i<5; i++){
            //Resample konfiguratu
            Resample rs = new Resample();
            rs.setInputFormat(datuak);
            rs.setSampleSizePercent(70);
            rs.setNoReplacement(true);
            rs.setBiasToUniformClass(1);
            rs.setSeed(i+1);

            Instances train = Filter.useFilter(datuak,rs);
            rs.setInvertSelection(true);
            Instances dev = Filter.useFilter(datuak,rs);
            //Preguntar que modelo tenemos que cargar
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(train);

            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(nb, dev);

            //Lortu avg balioak
            avgFmeasure = (eval.fMeasure(train.classIndex())+avgFmeasure)/(i+1);
            avgPrecission = (eval.precision(train.classIndex())+avgPrecission)/(i+1);
            avgRecall = (eval.recall(train.classIndex())+avgRecall)/(i+1);
        }
        //aqui hacer para que se escriba en 1 documento
    }

    private void fCV(Instances datuak) throws Exception{

        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(datuak);

        Evaluation eval = new Evaluation(datuak);
        eval.crossValidateModel(nb,datuak,10,new Random(1));

        Double fMeasure = eval.fMeasure(datuak.classIndex());
        Double precision = eval.precision(datuak.classIndex());
        Double recall = eval.recall(datuak.classIndex());

        //En un doku poner datos y el Confusion matrix
    }
}
