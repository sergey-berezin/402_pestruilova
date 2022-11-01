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
using System.Collections;

namespace NuGetEmotionFP
{
    public class EmotionFP : IDisposable
    {
        private InferenceSession session;
        private BufferBlock<ToTransform> input = new BufferBlock<ToTransform>();
        private TransformBlock<ToTransform, ToAction> ImgTransform;
        private ActionBlock<ToAction> ImgAction;
        private struct ToTransform
        {
            public Image<Rgb24> img;
            public BufferBlock<float[]> buf;
            public ToTransform(Image<Rgb24> i, BufferBlock<float[]> b)
            {
                img = i;
                buf = b;
            }
        }
        private struct ToAction
        {
            public List<NamedOnnxValue> list;
            public BufferBlock<float[]> buf;
            public ToAction (List<NamedOnnxValue> l, BufferBlock<float[]> b)
            {
                list = l;
                buf = b;
            }
        }

        public EmotionFP(CancellationToken token)
        {
            using var modelStream = typeof(EmotionFP).Assembly.GetManifestResourceStream("NuGetEmotionFP.emotion-ferplus-7.onnx");
            using var memoryStream = new MemoryStream();
            modelStream.CopyTo(memoryStream);
            this.session = new InferenceSession(memoryStream.ToArray());

            this.ImgTransform = new TransformBlock<ToTransform, ToAction>(async tr =>
             {
                 return await Task<ToAction>.Factory.StartNew(() =>
                 {
                     //Console.WriteLine("---Start");
                     token.ThrowIfCancellationRequested();
                     tr.img.Mutate(ctx => { ctx.Resize(new Size(64, 64)); });
                     var inp = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("Input3", GrayscaleImageToTensor(tr.img)) };
                     ToAction act = new ToAction(inp, tr.buf);
                     //Console.WriteLine("---Finish");
                     return act;
                 });
             }, new ExecutionDataflowBlockOptions()
             {
                 MaxDegreeOfParallelism = Environment.ProcessorCount,
                 CancellationToken = token
             });

 
            this.ImgAction = new ActionBlock<ToAction>(act =>
            {
                //Console.WriteLine("###Start");
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
                token.ThrowIfCancellationRequested();
                results = session.Run(act.list);
                float[] f = Softmax(results.First(v => v.Name == "Plus692_Output_0").AsEnumerable<float>().ToArray());
                act.buf.Post(f);
                //Console.WriteLine("###Finish");
                }, new ExecutionDataflowBlockOptions
            {
                MaxDegreeOfParallelism = 1,
                CancellationToken = token
            });
        }


        public void GetEmotions(Image<Rgb24> img, BufferBlock<float[]> bufferblock)
        {
            if (img == null)
            {
                float[] f = { 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F };
                bufferblock.Post(f);
                return;
            }
            try
            {
                ToTransform t = new ToTransform(img, bufferblock);
                input.Post(t);
                
                input.LinkTo(ImgTransform, new DataflowLinkOptions()
                {
                    PropagateCompletion = true
                });

                ImgTransform.LinkTo(ImgAction, new DataflowLinkOptions()
                {
                    PropagateCompletion = true
                });
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
            input.Complete();
            session.Dispose();
        }
        
    }
}


