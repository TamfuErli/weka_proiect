package Proiektua;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.stopwords.Rainbow;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class aurreProzesamendua {

    public static void main(String[] args){

        try{
            String csvTrain = args[0];
            String csvTrainMoldatua = args [1];
            String trainArff = args[2];
            Instances dataRaw = convertCSVtoARFF(csvTrain, csvTrainMoldatua);
            String hiztegi = args [3];
            String tarin_bow_fitx = args[4];
            String tarin_bow_fss_fitx = args[5];
            String hiztegiBerria = args[6];

            dataRaw.setClassIndex(1);
            Instances train_bow = createBagOfWords(dataRaw, hiztegi);
            fitxategiakGorde(train_bow, tarin_bow_fitx);
            seleccionFSS(train_bow, hiztegi, hiztegiBerria);


        } catch (Exception e){
            System.out.println(("ERROREA"));
            e.printStackTrace();
        }
    }


    public static Instances convertCSVtoARFF (String csvPath, String csvMoldatua)
        throws Exception {

        Path csv = Path.of(csvPath);
        String edukia = Files.readString(csv);
        String edukiaMoldatuta = edukia.replaceAll("http\\S+|@\\w+|#|\\W", "");

        FileWriter fw = new FileWriter(csvMoldatua);
        fw.write(edukiaMoldatuta);
        fw.close();

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvMoldatua));
        Instances data = loader.getDataSet();

        data.setClassIndex(1);
        Instances dataMoldatua = filterIrrelevant(data);

        dataMoldatua.renameAttribute(0, "gaiaAtr");
        dataMoldatua.renameAttribute(1, "konnotazioAtr");
        dataMoldatua.renameAttribute(2, "idAtr");
        dataMoldatua.renameAttribute(3, "dataAtr");
        dataMoldatua.renameAttribute(4, "textuAtr");

        return dataMoldatua;

    }

    public static Instances filterIrrelevant(Instances data) {
        Instances filteredData = new Instances(data);
        // Instantzzia guztietan
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            // Verificar si la clase NO es "irrelevant"
            if (!instance.stringValue(data.classIndex()).equals("irrelevant")) {
                filteredData.add(instance);  // Agregar instancia al nuevo conjunto
            }
        }
        return filteredData;
    }

    public static Instances createBagOfWords(Instances train_raw, String dictionaryPath)
            throws Exception{

        StringToWordVector bowFilter = new StringToWordVector();
        bowFilter.setDictionaryFileToSaveTo(new File(dictionaryPath));
        bowFilter.setWordsToKeep(5000);
        bowFilter.setLowerCaseTokens(true);
        bowFilter.setOutputWordCounts(true);
        bowFilter.setStopwordsHandler(new Rainbow());
        bowFilter.setAttributeIndices("4");

        bowFilter.setInputFormat(train_raw);
        Instances train_bow = Filter.useFilter(train_raw, bowFilter);


        return train_bow;
    }

    public static void seleccionFSS(Instances train_bow, String hiztegia, String hiztegiHobetua) throws Exception{
        AttributeSelection filter = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(2000);

        filter.setEvaluator(eval);
        filter.setSearch(ranker);
        filter.setInputFormat(train_bow);

        Instances train_bow_fss = Filter.useFilter(train_bow, filter);

        FileWriter fw = new FileWriter(hiztegiHobetua);
        try{
            BufferedReader reader = new BufferedReader(new FileReader(hiztegia));
            reader.readLine();
            for (int i = 0; i < train_bow_fss.numAttributes() - 1; i++) {
                String atr = train_bow_fss.attribute(i).name();
                String line;
                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                    String atrName = line.split(",")[0];
                    if (atrName.equals(atr)) {
                        System.out.println("badago: " + atr);
                        fw.write(line + "\n");

                    }
                }
                reader = new BufferedReader(new FileReader(hiztegia));
            }
        } catch (Exception e) {
            System.out.println("ERROREA hiztegi berria lortzean");
            e.printStackTrace();
        }
        fw.close();
       fitxategiakGorde(train_bow_fss, hiztegiHobetua);
    }

    public static Instances totalFSS(Instances dataRaw, String hiztegia) throws Exception{
        FixedDictionaryStringToWordVector filtro = new FixedDictionaryStringToWordVector();
        filtro.setDictionaryFile(new File(hiztegia));
        filtro.setInputFormat(dataRaw);

        Instances filteredData_bow_fss = Filter.useFilter(dataRaw, filtro);

        return filteredData_bow_fss;

    }

    public static void fitxategiakGorde(Instances data, String fitxategia) throws Exception{
        ArffSaver dataSaver = new ArffSaver();
        dataSaver.setFile(new File(fitxategia));
        dataSaver.setInstances(data);
        dataSaver.writeBatch();
    }
}
