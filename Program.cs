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

EmotionFP em = new EmotionFP();

var buffer = new BufferBlock<float[]>();

CancellationTokenSource source = new CancellationTokenSource();
CancellationToken token = source.Token;

string[] keys = { "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt" };

Console.WriteLine("\n############-Test1-############");

em.GE(image1, buffer, token);
float[] emotions = buffer.Receive();
Console.WriteLine("\n---face1---");
foreach (var i in keys.Zip(emotions))
    Console.WriteLine($"{i.First}: {i.Second}");


em.GE(image2, buffer, token);
float[] emotions2 = buffer.Receive();
Console.WriteLine("\n---face2---");
foreach (var i in keys.Zip(emotions2))
    Console.WriteLine($"{i.First}: {i.Second}");

Console.WriteLine("\n############-Test2-############");

var buf = new BufferBlock<float[]>();

Parallel.For(0, 4, i => {
    em.GE(img[i], buffer, token);
    float[] emotions = buffer.Receive();
    Console.WriteLine($"\n---face{i+1}---");
    foreach (var j in keys.Zip(emotions))
        Console.WriteLine($"{j.First}: {j.Second}");
});

Console.WriteLine("\n############-Test3-############");

Image<Rgb24>? image0 = null;
em.GE(image0, buffer, token);
float[] emotions3 = buffer.Receive();
Console.WriteLine("\n---face0---");
foreach (var i in keys.Zip(emotions3))
    Console.WriteLine($"{i.First}: {i.Second}");
