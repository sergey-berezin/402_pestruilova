using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using static System.Net.Mime.MediaTypeNames;
using Image = SixLabors.ImageSharp.Image;
using System.Threading.Tasks.Dataflow;

namespace NuGetEmotionFP
{
    public class EmotionFP : IDisposable
    {
        private InferenceSession session;
        private BufferBlock<Image<Rgb24>> input = new BufferBlock<Image<Rgb24>>();
        private object obj = new object();

        public EmotionFP()
        {
            using var modelStream = typeof(EmotionFP).Assembly.GetManifestResourceStream("NuGetEmotionFP.emotion-ferplus-7.onnx");
            using var memoryStream = new MemoryStream();
            modelStream.CopyTo(memoryStream);
            this.session = new InferenceSession(memoryStream.ToArray());
            /*try
            {
                using var modelStream = typeof(EmotionFP).Assembly.GetManifestResourceStream("NuGetEmotionFP.emotion-ferplus-7.onnx");
                using var memoryStream = new MemoryStream();
                modelStream.CopyTo(memoryStream);
                this.session = new InferenceSession(memoryStream.ToArray());
            }
            catch(Exception e)
            {
                if (e.Source != null)
                    Console.WriteLine("IOException source: {0}", e.Message);
                else Console.WriteLine("-");
            }*/
        }


        public void GE(Image<Rgb24> img, BufferBlock<float[]> bufferblock, CancellationToken token)
        {
            if (img == null)
            {
                float[] f = { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F };
                bufferblock.Post(f);
                return;
            }
            try
            {
                var input = new BufferBlock<Image<Rgb24>>();
                input.Post(img);
                var t1 = new TransformBlock<Image<Rgb24>, List<NamedOnnxValue>>(async img =>
                {
                    return await Task<List<NamedOnnxValue>>.Factory.StartNew(() =>
                        {
                            token.ThrowIfCancellationRequested();
                            img.Mutate(ctx => { ctx.Resize(new Size(64, 64)); });
                            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("Input3", GrayscaleImageToTensor(img)) };
                            return inputs;
                        });
                }, new ExecutionDataflowBlockOptions()
                {
                    MaxDegreeOfParallelism = Environment.ProcessorCount,
                    CancellationToken = token
                });


                var t2 = new ActionBlock<List<NamedOnnxValue>>(async inputs =>
                {
                    await Task.Factory.StartNew(() =>
                    {
                        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
                        token.ThrowIfCancellationRequested();
                        lock (obj)
                        {
                            results = session.Run(inputs);
                        }

                        float[] f = results.First(v => v.Name == "Plus692_Output_0").AsEnumerable<float>().ToArray();
                        bufferblock.Post(f);
                    });
                }, new ExecutionDataflowBlockOptions
                {
                    MaxDegreeOfParallelism = 1,
                    CancellationToken = token
                });

                input.LinkTo(t1, new DataflowLinkOptions()
                {
                    PropagateCompletion = true
                });

                t1.LinkTo(t2, new DataflowLinkOptions()
                {
                    PropagateCompletion = true
                });

                input.Complete();
            }
            catch (Exception e)
            {
                //Console.WriteLine("Exception: ", e.Message);
                float[] f = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
                bufferblock.Post(f);
            }

        }

        static DenseTensor<float> GrayscaleImageToTensor(Image<Rgb24> img)
        {
            var w = img.Width;
            var h = img.Height;
            var t = new DenseTensor<float>(new[] { 1, 1, h, w });

            img.ProcessPixelRows(pa =>
            {
                for (int y = 0; y < h; y++)
                {
                    Span<Rgb24> pixelSpan = pa.GetRowSpan(y);
                    for (int x = 0; x < w; x++)
                    {
                        t[0, 0, y, x] = pixelSpan[x].R;
                    }
                }
            });

            return t;
        }

        public float[] Softmax(float[] z)
        {
            var exps = z.Select(x => Math.Exp(x)).ToArray();
            var sum = exps.Sum();
            return exps.Select(x => (float)(x / sum)).ToArray();
        }


        public void Dispose()
        {
            session.Dispose();
        }
        
    }
}


