import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * @author Harrison Chen does the pel array stuff, inputs are buffered so its fast (go to the errors and fix the line, then comment out/delete)
 */
public class EditByteFiles
{
   public static void main(String[] args) throws FileNotFoundException, IOException
   {
      int width = 3072;
      int height = 4080; 
      int w = 0, h = 0;
      int[][] image = new int[height][width];
      byte[] bytes = new byte[width*height];

      for (int set = 1; set <= 6; set++)
      {
         for (int finger = 1; finger <= 5; finger++)
         {
            System.out.println("Editing "+finger+set+"");
            File imageByteFile = new File("./byte_files/"+finger+set+".bin"); 
            DataInputStream in = new DataInputStream(new FileInputStream(imageByteFile));

            in.read(bytes);
            in.close();
            in = new DataInputStream(new ByteArrayInputStream(bytes));

            //int index = 0;
            for (int i = 0; i < height; i++)
            {
               for (int j = 0; j < width; j++)
               {
                  image[i][j] = (int) in.readByte();
               }
            }

            PelArray pArr = new PelArray(image);
            //pArr = pArr.onesComplimentImage();

            pArr = pArr.offsetColors(0, -40, -10);

            if (pArr.calcCOM())
            {
               int x = pArr.getXcom();
               int y = pArr.getYcom();
               pArr = pArr.crop(x-1330,y-1000,x+1300,y+2000); 
            }

            h = pArr.getHeight()/20;
            w = pArr.getWidth()/20;

            int[][] scaled = pArr.scale(w,h).getPelArray();

            File outputFile = new File("./edited_byte_files/"+finger+set+".bin"); 
            DataOutputStream out = new DataOutputStream(new FileOutputStream(outputFile));

            for (int i = 0; i < h; i++)
               for (int j = 0; j < w; j++)
                  out.writeByte((byte) (scaled[i][j]));
            
            out.close();
         }
      }

      System.out.println("h:"+h+", w:"+w);
   }
}