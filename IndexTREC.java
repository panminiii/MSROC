package sigir;

import java.io.*;
import java.util.*;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;
import org.apache.lucene.benchmark.byTask.feeds.*;
import org.apache.lucene.benchmark.byTask.utils.*;
import org.apache.lucene.document.*;


public class IndexTREC {

	private IndexTREC() {
	}

	public static void main(String[] args) throws Exception {

		String usage = "java org.apache.lucene.demo.IndexFiles"
				+ " [-index INDEX_PATH] [-docs DOCS_PATH] [-update]\n\n"
				+ "This indexes the documents in DOCS_PATH, creating a Lucene index"
				+ "in INDEX_PATH that can be searched with SearchFiles";
		String indexPath = "index_test_1";
		String docsPath = "E:\\AP11\\";
//		String indexPath = "index_AP901";
//		String docsPath = "F:/TREC_DATA/collection/AP/AP90";
//		String indexPath = "index_AP8889";
//		String docsPath = "F:/TREC_DATA/collection/AP/AP8889";
//		String indexPath = "index_disk12";
//		String docsPath = "F:/TREC_DATA/collection/disk12";
//		String indexPath = "index_disk45";
//		String docsPath = "F:/TREC_DATA/collection/disk45";
//		String indexPath = "index_wt2g";
//		String docsPath = "F:/TREC_DATA/collection/wt2g";
//		String indexPath = "index_wt10g";
//		String docsPath = "F:/TREC_DATA/collection/wt10g";
//		String indexPath = "index_SJMN";
//		String docsPath = "F:/TREC_DATA/collection/disk3/SJMN";
//		String indexPath = "index_WSJ";
//		String docsPath = "F:/TREC_DATA/collection/disk12/WSJ";
//		String indexPath = "index_FBIS";
//		String docsPath = "F:/TREC_DATA/collection/disk45/FBIS";
//		String indexPath = "index_FT";
//		String docsPath = "F:/TREC_DATA/collection/disk45/FT";
//		String indexPath = "index_AP";
//		String docsPath = "F:/TREC_DATA/collection/AP";
//		String indexPath = "index_LA";
//		String docsPath = "F:/TREC_DATA/collection/disk45/LA";
	//	String indexPath = "index_GOV2";
	//	String docsPath = "F:/TREC_DATA/collection/GOV2";


		boolean create = true;
		for(int i=0;i<args.length;i++) {
			if ("-index".equals(args[i])) {
				indexPath = args[i+1];
				i++;
			} else if ("-docs".equals(args[i])) {
				docsPath = args[i+1];
				i++;
			} else if ("-update".equals(args[i])) {
				create = false;
			}
		}

		if (docsPath == null) {
			System.err.println("Usage: " + usage);
			System.exit(1);
		}

		final File docDir = new File(docsPath);
		if (!docDir.exists() || !docDir.canRead()) {
			System.out.println("Document directory '" +docDir.getAbsolutePath()+ "' does not exist or is not readable, please check the path");
			System.exit(1);
		}

		Date start = new Date();
		try {
			System.out.println("Indexing to directory '" + indexPath + "'...");

			Directory dir = FSDirectory.open(new File(indexPath));
			//Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_41);
			Analyzer analyzer = new MyStopAndStemmingAnalyzer();
			IndexWriterConfig iwc = new IndexWriterConfig(Version.LUCENE_41,analyzer);
			iwc.setOpenMode(OpenMode.CREATE);
			iwc.setRAMBufferSizeMB(1024.0);

			IndexWriter writer = new IndexWriter(dir, iwc);

			TrecContentSource tcs = new TrecContentSource();
			Properties props = new Properties();
			props.setProperty("print.props", "false");
			props.setProperty("content.source.verbose", "false");
			props.setProperty("content.source.excludeIteration", "true");
			props.setProperty("doc.maker.forever", "false");
			props.setProperty("work.dir", ".");//
			props.setProperty("docs.dir", docsPath);// 
			props.setProperty("trec.doc.parser", sigir.myTrecParser.class.getName());
			props.setProperty("content.source.forever", "false");
			tcs.setConfig(new Config(props));
			tcs.resetInputs();
			int n = 0;
			DocData dd = new DocData();
		
			while (true) {
				try {
					dd = tcs.getNextDocData(dd);
					//n++;
					//System.out.println(n);
				} catch (NoMoreDataException e) {
					//n++;
					System.err.println(n);
					break;
				}

				Document doc = new Document();
				doc.add(new StringField("docno", dd.getName(), Field.Store.YES));
				System.out.println(dd.getName());
				FieldType textWithTermVectors = new FieldType(TextField.TYPE_STORED);
				textWithTermVectors.setIndexed(true);
				textWithTermVectors.setStoreTermVectors(true);

				doc.add(new Field("contents", dd.getBody(), textWithTermVectors));
				//doc.add(new Field("head", dd.getTitle(), textWithTermVectors));

				
				if(writeProcessedDocsToFile(analyzer, dd.getBody())){

					writer.addDocument(doc);
					n++;
					System.out.println(n);
				}
			}

			writer.forceMerge(1);
			writer.close();
			tcs.close();
			Date end = new Date();
			System.out.println(end.getTime() - start.getTime() + " total milliseconds");
		}catch (IOException e) {
			System.out.println(" caught a " + e.getClass() +
					"\n with message: " + e.getMessage());
		}
	}

	/**
	 * 
	 * @param writer			
	 * @param analyzer		
	 * @param text			
	 * @throws IOException
	 */
	public static boolean writeProcessedDocsToFile(Analyzer analyzer, String text) throws IOException {

		ArrayList<String> terms = MyStopAndStemmingAnalyzer.getTermList(analyzer, text);
		if (terms.size() == 0) {
			//writer.write("\r\n");

			return false;
		}
		return true;
	}
}