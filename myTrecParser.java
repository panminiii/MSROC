package sigir;

import java.io.IOException;

import org.apache.lucene.benchmark.byTask.feeds.DocData;
import org.apache.lucene.benchmark.byTask.feeds.TrecContentSource;
import org.apache.lucene.benchmark.byTask.feeds.TrecDocParser;


public class myTrecParser extends TrecDocParser {

 
  //private static final String DOCNO = "<DOCNO>";
  //private static final String DOCNO_END = "</DOCNO>";

 
  
  @Override
  public DocData parse(DocData docData, String name, TrecContentSource trecSrc, 
      StringBuilder docBuf, ParsePathType pathType) throws IOException {
    int mark = 0; // that much is skipped

    /*
    String id = extract(docBuf, DOCNO, DOCNO_END, -1, null);
    if (id!=null) {
      id = stripTags(id,0).toString().trim();
    }
    */

    docData.clear();
    docData.setName(name);

    
    String content1 = docBuf.toString().replaceAll("<DOCHDR>[\u0000-\uFFFF]+</DOCHDR>","");
//    String content3 = content1.replaceAll("<DDOCOLDNO>[\u0000-\uFFFF]+</DOCOLDNO>","");
    String content2 = stripTags(content1, mark).toString();
    docData.setBody(content2);

    //System.out.println(docBuf.length() +"\t" + content1.length() + "\t" + content2.length() );
   
//    System.out.println(content2);
    
    return docData;
  }

}
