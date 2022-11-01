using NuGetEmotionFP;
using System.Threading.Tasks.Dataflow;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;


using Image<Rgb24> image1 = Image.Load<Rgb24>("face1.png");
using Image<Rgb24> image2 = Image.Load<Rgb24>("face2.png");
using Image<Rgb24> image3 = Image.Load<Rgb24>("face3.png");
using Image<Rgb24> image4 = Image.Load<Rgb24>("face4.png");
Image<Rgb24>[] img = new Image<Rgb24>[4];
img[0] = image1;
img[1] = image2;
img[2] = image3;
img[3] = image4;

CancellationTokenSource source = new CancellationTokenSource();
CancellationToken token = source.Token;

EmotionFP em = new EmotionFP(token);

string[] keys = { "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt" };

Console.WriteLine("\n############-Test1-############");

var buffer1_1 = new BufferBlock<float[]>();
em.GetEmotions(image1, buffer1_1);
float[] emotions1 = buffer1_1.Receive();
Console.WriteLine("\n---face1---");
foreach (var i in keys.Zip(emotions1))
    Console.WriteLine($"{i.First}: {i.Second}");

var buffer1_2 = new BufferBlock<float[]>();
em.GetEmotions(image2, buffer1_2);
float[] emotions2 = buffer1_2.Receive();
Console.WriteLine("\n---face2---");
foreach (var i in keys.Zip(emotions2))
    Console.WriteLine($"{i.First}: {i.Second}");

Console.WriteLine("\n############-Test2-############");

BufferBlock<float[]>[] buffer2 = new BufferBlock<float[]>[4];
for(int i = 0; i < 4; i++)
{
    buffer2[i] = new BufferBlock<float[]>();
}
Parallel.For(0, 4, i => {
    
    em.GetEmotions(img[i], buffer2[i]);
});
for (int i = 0; i < 4; i++)
{
    float[] emotions = buffer2[i].Receive();
    Console.WriteLine($"\n---face{i + 1}---");
    foreach (var j in keys.Zip(emotions))
        Console.WriteLine($"{j.First}: {j.Second}");
}

Console.WriteLine("\n############-Test3-############");

Image<Rgb24>? image0 = null;
var buffer3 = new BufferBlock<float[]>();
em.GetEmotions(image0, buffer3);
float[] emotions3 = buffer3.Receive();
Console.WriteLine("\n---face0---");
foreach (var i in keys.Zip(emotions3))
    Console.WriteLine($"{i.First}: {i.Second}");
