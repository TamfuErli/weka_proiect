package Proiektua;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.stopwords.Rainbow;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class aurreProzesamendua {

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

        data.setClassIndex(data.numAttributes()-1);
        return data;

    }

    public static Instances filterIrrelevant(Instances data) {
        Instances filteredData = new Instances(data);
        filteredData.delete();
        for (int i = 0; i < data.numInstances(); i++){
            if(!data.instance(i).stringValue(data.classIndex()).equals("irrelevant")){
                filteredData.add(data.instance(i));
            }
        }
        return filteredData;
    }

    public static Instances createBagOfWords(Instances tweets, String dictionaryPath)
            throws Exception{

        StringToWordVector bowFilter = new StringToWordVector();
        bowFilter.setDictionaryFileToSaveTo(new File(dictionaryPath));
        bowFilter.setWordsToKeep(5000);
        bowFilter.setLowerCaseTokens(true);
        bowFilter.setOutputWordCounts(true);
        bowFilter.setStopwordsHandler(new Rainbow());
        bowFilter.setAttributeIndices("4");

        bowFilter.setInputFormat(tweets);
        Instances tweetsVectorized = Filter.useFilter(tweets, bowFilter);

        System.out.println("[DEBUG] Vocabulario creado: " + tweetsVectorized.numAttributes() + " términos");
        System.out.println("[DEBUG] Ejemplo de vector: " + tweetsVectorized.firstInstance());

        return tweetsVectorized;
    }

    public static Instances seleccionFSS(Instances data, String outputDictionary) throws Exception{
        AttributeSelection filter = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(2000);

        filter.setEvaluator(eval);
        filter.setSearch(ranker);
        filter.setInputFormat(data);

        Instances filteredData = Filter.useFilter(data, filter);

        FileWriter fw = new FileWriter(outputDictionary);
        for (int i = 0; i < filteredData.numAttributes()-1; i++){
            fw.write(data.attribute(i).name()+ "/n");
        }
        fw.close();

        return filteredData;
    }
}
