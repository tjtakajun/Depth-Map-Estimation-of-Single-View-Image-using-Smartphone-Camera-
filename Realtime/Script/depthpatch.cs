using UnityEngine;
using UnityEngine.UI;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;

public class depthpatch : MonoBehaviour
{
    public RawImage rawImage;
    private void Update()
    {

    }
    public void main(
        Texture2D modeldepth,
        Texture2D modelimage,
        int patch_size,//12
        string output,
        bool plotprogress,
        Texture2D Image) {
        //var args = parse_args();
        UnityEngine.Debug.Log("modelimage");
        //var image = ToBitmap(Image);//←切り取った後の画像格納
        var inpainter = new Inpainter(modelimage, modeldepth, Image, patch_size, plotprogress, rawImage);
        var output_image = inpainter.inpaint();
        //var med = new Median(output_image);
        //output_image = med.median();
        Texture2D op = createReadabeTexture2D(output_image);
        byte[] bytes = op.EncodeToPNG();
        File.WriteAllBytes("/storage/emulated/0/DCIM/Camera/output.png", bytes);
    }

    Texture2D createReadabeTexture2D(Texture2D texture2d)
    {
        RenderTexture renderTexture = RenderTexture.GetTemporary(
                    texture2d.width,
                    texture2d.height,
                    0,
                    RenderTextureFormat.Default,
                    RenderTextureReadWrite.Linear);

        Graphics.Blit(texture2d, renderTexture);
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = renderTexture;
        Texture2D readableTextur2D = new Texture2D(texture2d.width, texture2d.height);
        readableTextur2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        readableTextur2D.Apply();
        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(renderTexture);
        return readableTextur2D;
    }

    /*public static Bitmap ToBitmap(Texture2D texture)//Texture2Dをビットマップに変換
    {
        // Texture2Dからピクセルデータを取得
        Color32[] pixels = texture.GetPixels32();

        // Bitmapを生成
        Bitmap bitmap = new Bitmap(texture.width, texture.height);

        // ピクセルデータをBitmapにセット
        for (int y = 0; y < texture.height; y++)
        {
            for (int x = 0; x < texture.width; x++)
            {
                Color32 color = pixels[y * texture.width + x];
                bitmap.SetPixel(x, y, Color.FromArgb(color.a, color.r, color.g, color.b));
            }
        }

        return bitmap;
    }*/

    /*public static void parse_args() {
        var parser = argparse.ArgumentParser();
        parser.add_argument("-ps", "--patch_size", help: "the size of the patches", type: @int, @default: 9);
        parser.add_argument("-o", "--output", help: "the file path to save the output image", @default: "result/output9 random third.png");
        parser.add_argument("--plot-progress", help: "plot each generated image", action: "store_true", @default: false);
        parser.add_argument("model_image", help: "the model image");
        parser.add_argument("model_depth", help: "the model depth of model image");
        parser.add_argument("image", help: "the image you want to make depth image with");
        return parser.parse_args();
    }*/

    public class Inpainter {

        private double[,] confidence;

        private double[,] data;

        private uint[,] front;

        private uint[,,] model_depth;
        private uint[,,] model_image;
        private uint[,,] image;

        private int patch_size;
        private uint[,,] output_image;
        private bool plot_progress;

        double[,] priority;

        uint[,,] working_image;

        uint[,] working_mask;

        RawImage raw;

        public Inpainter(
            Texture2D modelimage,
            Texture2D modeldepth,
            Texture2D Image,
            int patchsize,
            bool plotprogress,
            RawImage rawImage) {
            this.model_image = Texture2DToUintArray3D(modelimage);//←モデル画像格納
            this.model_depth = Texture2DToUintArray3D(modeldepth);//←深度モデル格納
            this.image = Texture2DToUintArray3D(Image);//←切り取った後の画像格納 3次元用も必要
            this.patch_size = patchsize;
            this.plot_progress = plotprogress;
            // Non initialized attributes
            this.working_image = null;
            this.front = null;
            this.data = null;
            this.priority = null;
            this.raw = rawImage;
        }

        //  Compute the new image and return it 
        public Texture2D inpaint() {
            this._validate_inputs();//
            this._initialize_attributes();//

            var start_time = DateTime.Now;
            var keep_going = true;
            while (keep_going) {
                this._find_front();
                if (this.plot_progress) {
                    //←推定の様子を示すやつ
                    this._plot_image();
                }
                var priority_start_time = DateTime.Now;
                this._update_priority();
                UnityEngine.Debug.Log(String.Format("_update_priority: {0} seconds", DateTime.Now - priority_start_time));
                int[] target_pixel = new int[] { _find_highest_priority_pixel().maxY, _find_highest_priority_pixel().maxX };
                var find_start_time = DateTime.Now;
                var source_patch = this._find_source_patch(target_pixel);
                UnityEngine.Debug.Log(String.Format("Time to find best: {0} seconds", DateTime.Now - find_start_time));
                this._update_image(target_pixel, source_patch);
                keep_going = !this._finished();
            }
            UnityEngine.Debug.Log(String.Format("Took {0} seconds to complete", DateTime.Now - start_time));

            int height = output_image.GetLength(0);
            int width = output_image.GetLength(1);
            /*Texture2D texture = new Texture2D(height, width);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    Color color = new Color((float)output_image[i, j, 0] / 255f, (float)output_image[i, j, 1] / 255f, (float)output_image[i, j, 2] / 255f);
                    texture.SetPixel(j, i, color);
                }
            }*/

            Texture2D texture = new Texture2D(width, height);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    byte r = (byte)(output_image[i, j, 0]);
                    byte g = (byte)(output_image[i, j, 1]);
                    byte b = (byte)(output_image[i, j, 2]);
                    Color color = new Color32(r, g, b, 255);
                    texture.SetPixel(j, i, color);
                }
            }
            texture.Apply();
            return texture;//uint[,,]からTexture2Dに変換
        }

        public virtual void _validate_inputs() { //入力のサイズが合っているかどうか，完成
            if (this.model_image.GetLength(1) != this.model_depth.GetLength(1) || this.model_image.GetLength(0) != this.model_depth.GetLength(0)) {
                throw new Exception("mask and image must be of the same size");
            }
        }

        //  Initialize the non initialized attributes
        // 
        //         The confidence is initially the inverse of the mask, that is, the
        //         target region is 0 and source region is 1.
        // 
        //         The data starts with zero for all pixels.
        // 
        //         The working image and working mask start as copies of the original
        //         image and mask.
        //         
        public void _initialize_attributes() {//作成

            // 非初期化属性を初期化
            //int model_height = model_image.GetLength(0);
            //int model_width = model_image.GetLength(1);
            int height = image.GetLength(0);
            int width = image.GetLength(1);

            this.priority = new double[height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    this.priority[y, x] = 0;
                }
            }

            this.data = new double[height, width];
            //this.output_image = new int[height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    this.data[y, x] = 0;
                }
            }

            // imageのピクセルデータをuint[,,]に格納
            this.working_image = new uint[height, width,3];
            this.output_image = new uint[height, width,3];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    for (int k=0; k<3; k++) 
                    {
                        this.working_image[y, x, k] = image[y, x, k];
                        this.output_image[y, x, k] = 0;
                    }
                    
                }
            }

            this.working_mask = new uint[height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    this.working_mask[y, x] = 1;
                }
            }
            for (int y = 0; y < height; y++)
            {
                this.working_mask[y, 0] = 0;
                this.working_mask[y, width - 1] = 0;
            }
            for (int x = 0; x < width; x++)
            {
                this.working_mask[0, x] = 0;
                this.working_mask[height - 1, x] = 0;
            }
            this.confidence = new double[height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    this.confidence[y, x] = (double)(1 - this.working_mask[y, x]);
                }
            }
        }


        //  Find the front using laplacian on the mask
        // 
        //         The laplacian will give us the edges of the mask, it will be positive
        //         at the higher region (white) and negative at the lower region (black).
        //         We only want the white region, which is inside the mask, so we
        //         filter the negative values.
        //         
        public virtual void _find_front() {
            int width = this.working_mask.GetLength(1);
            int height = this.working_mask.GetLength(0);
            var Laplace_res = Laplace(this.working_mask);
            this.front = new uint[height, width];
            for (int i = 1; i < height - 1; i++)
            {
                for (int j = 1; j < width - 1; j++)
                {
                    if (Laplace_res[i,j] > 0){
                         this.front[i,j] = 1;
                    }
                    else
                    {
                        this.front[i, j] = 0;
                    }
                }
            }
            // self.front = (laplace_numpy(self.working_mask) > 0).astype('uint8')
            // self.front = (laplace_numba(self) > 0).astype('uint8')
            // TODO: check if scipy's laplace filter is faster than scikit's
        }

        // Laplaceフィルタを適用するメソッド
        static int[,] Laplace(uint[,] image)//working_maskに適用
        {
            int width = image.GetLength(1);
            int height = image.GetLength(0);

            int[,] result = new int[height, width];

            for (int i = 1; i < height - 1; i++)
            {
                for (int j = 1; j < width - 1; j++)
                {
                    // 周りの3x3マスを取得する
                    int sum =
                    (int)image[i - 1, j - 1] + (int)image[i - 1, j] + (int)image[i - 1, j + 1] +
                    (int)image[i, j - 1] + (int)image[i, j] * (-8) + (int)image[i, j + 1] +
                    (int)image[i + 1, j - 1] + (int)image[i + 1, j] + (int)image[i + 1, j + 1];

                    result[i, j] = sum > 0 ? 1 : 0;
                }
            }
            return result;
        }

        //public RawImage rawImage;
        //public RawImage rawImage;
        public void _plot_image()
        {
            if (this.raw == null)
            {
                UnityEngine.Debug.Log("Raw is null.");
            }
            Destroy(this.raw.texture, 5f);
            int height = this.working_mask.GetLength(0);
            int width = this.working_mask.GetLength(1);
            // Remove the target region from the image
            uint[,] inverse_mask = new uint[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    inverse_mask[i, j] = 1 - this.working_mask[i, j];
                }
            }
            uint[,,] rgb_inverse_mask = _to_rgb(inverse_mask);

            uint[,,] image = new uint[height, width, 3];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        image[i, j, k] = (byte)(this.output_image[i, j, k] * rgb_inverse_mask[i, j, k]);
                    }
                }
            }

            // Fill the target borders with red
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    image[i, j, 0] += (byte)(this.front[i, j] * 255);
                }
            }

            // Fill the inside of the target region with white
            uint[,] white_region = new uint[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        output_image[i, j, k] *= inverse_mask[i, j];
                    }
                }
            }
            // Fill the target borders with red
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    output_image[i, j, 0] += this.front[i, j] * 255;
                }
            }

            // Fill the inside of the target region with white
            uint[,] whiteRegion = new uint[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    whiteRegion[i, j] = (byte)((this.working_mask[i, j] - this.front[i, j]) * 255);
                }
            }
            var rgbWhiteRegion = _to_rgb(whiteRegion);

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        output_image[i, j, k] += rgbWhiteRegion[i, j, k];
                    }
                }
            }
            var texture = new Texture2D(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color color = new Color(output_image[y, x, 0], output_image[y, x, 1], output_image[y, x, 2]);
                    texture.SetPixel(x, y, color);
                }
            }
            texture.Apply();
            this.raw.enabled = true;
            this.raw.texture = texture;
        }


        public void _update_priority() {
            this._update_confidence();
            this._update_data();

            int width = this.confidence.GetLength(1);
            int height = this.confidence.GetLength(0);

            if (this.confidence != null)
            {
                UnityEngine.Debug.Log("confidence is available.");
            }

            if (this.data != null)
            {
                UnityEngine.Debug.Log("data is available.");
            }

            if (this.front != null)
            {
                UnityEngine.Debug.Log("front is available.");
            }


            var C = this.confidence;
            var D = this.data;
            var F = this.front;

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    this.priority[i,j] = C[i,j] * D[i,j] * (double)F[i,j];
                }
            }
        }

        public void _update_confidence() {
            var new_confidence = this.confidence.Clone() as double[,];
            List<int[]> front_positions = new List<int[]>();
            for (int i = 0; i < this.front.GetLength(0); i++)
            {
                for (int j = 0; j < this.front.GetLength(1); j++)
                {
                    if (this.front[i, j] == 1)
                    {
                        int[] position = new int[] { i, j };
                        front_positions.Add(position);
                    }
                }
            }

            foreach (var point in front_positions) {
                var patch = this._get_patch(point);
                new_confidence[point[0], point[1]] = Sum(_patch_data2D(this.confidence, patch)) / _patch_area(patch);
            }
            this.confidence = new_confidence;
        }

        public static double Sum(double[,] data)
        {
            int height = data.GetLength(0);
            int width = data.GetLength(1);
            double sum = 0;

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    sum += data[i, j];
                }
            }
            return sum;
        }

        public List<int[]> _get_patch(int[] point) {
            var half_patch_size = (this.patch_size - 1) / 2;
            var height = this.working_image.GetLength(0);
            var width = this.working_image.GetLength(1);
            var patch = new List<int[]> {
                new int[] {
                    Math.Max(0, point[0] - half_patch_size),
                    Math.Min(point[0] + half_patch_size, height - 1)
                },
                new int[] {
                    Math.Max(0, point[1] - half_patch_size),
                    Math.Min(point[1] + half_patch_size, width - 1)
                }
            };
            return patch;
        }

        public static int _patch_area(List<int[]> patch) {
            return (1 + patch[0][1] - patch[0][0]) * (1 + patch[1][1] - patch[1][0]);
        }

        public void _update_data()
        {
            var no = this._calc_normal_matrix();
            var gr = this._calc_gradient_matrix();

            // 2次元配列の要素の積を計算
            double[,,] normal_gradient = new double[no.GetLength(0), no.GetLength(1),no.GetLength(2)];
            for (int i = 0; i < no.GetLength(0); i++)
            {
                for (int j = 0; j < no.GetLength(1); j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        normal_gradient[i, j, k] = no[i, j, k] * gr[i, j, k];
                    }
                }
            }

            for (int i = 0; i < no.GetLength(0); i++)
            {
                for (int j = 0; j < no.GetLength(1); j++)
                {
                        this.data[i,j] = Math.Sqrt(Math.Pow(normal_gradient[i, j, 0], 2) + Math.Pow(normal_gradient[i, j, 1], 2));//縦横勾配の三平方の定理
                }
            }
        }

        public double[,,] _calc_normal_matrix() {
            // seikigyouretu no keisan
            // 計算用のカーネルを定義
            double[,] xKernel = { { 0.25, 0, -0.25 }, { 0.5, 0, -0.5 }, { 0.25, 0, -0.25 } };
            double[,] yKernel = { { -0.25, -0.5, -0.25 }, { 0, 0, 0 }, { 0.25, 0.5, 0.25 } };

            // カーネルを使ってコンボリューションを計算
            var xNormal = Convolve(this.working_mask, xKernel);
            var yNormal = Convolve(this.working_mask, yKernel);

            // コンボリューション結果を連結
            var normal = new double[xNormal.GetLength(0), xNormal.GetLength(1), 2];
            for (int i = 0; i < xNormal.GetLength(0); i++)
            {
                for (int j = 0; j < xNormal.GetLength(1); j++)
                {
                    normal[i, j, 0] = xNormal[i, j];
                    normal[i, j, 1] = yNormal[i, j];
                }
            }

            // 正規化
            var height = normal.GetLength(0);
            var width = normal.GetLength(1);
            var yn = yNormal.Cast<double>().ToArray();
            var xn = xNormal.Cast<double>().ToArray();
            var norm = Enumerable.Range(0, height * width).Select(k => Math.Sqrt(Math.Pow(yn[k], 2) + Math.Pow(xn[k], 2))).ToArray();
            norm = norm.Select(x => x == 0 ? 1 : x).ToArray();//xが0なら1を代入
            var unit_normal = new double[height, width, 2];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    unit_normal[i, j, 0] = normal[i, j, 0] / norm[i * width + j];
                    unit_normal[i, j, 1] = normal[i, j, 1] / norm[i * width + j];
                }
            }

            return unit_normal;
        }

        public double[,] Convolve(uint[,] image, double[,] kernel)//畳み込み積分
        {
            int kernelWidth = kernel.GetLength(1);
            int kernelHeight = kernel.GetLength(0);
            int imageWidth = image.GetLength(1);
            int imageHeight = image.GetLength(0);
            double[,] result = new double[imageHeight, imageWidth];
            for (int i = 0; i < imageHeight; i++)
            {
                for (int j = 0; j < imageWidth; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < kernelHeight; k++)
                    {
                        for (int l = 0; l < kernelWidth; l++)
                        {
                            int x = i + k - kernelHeight / 2;
                            int y = j + l - kernelWidth / 2;
                            if (x >= 0 && x < imageHeight && y >= 0 && y < imageWidth)
                            {
                                sum += (double)image[x, y] * kernel[k, l];
                            }
                        }
                    }
                    result[i, j] = sum;
                }
            }
            return result;
        }


        public double[,,] _calc_gradient_matrix() {
            // TODO: find a better method to calc the gradient
            int height = this.working_image.GetLength(0);
            int width = this.working_image.GetLength(1);
            var grey_image = rgb2gray(this.working_image);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (this.working_mask[i, j] == 1)
                    {
                        grey_image[i, j] = 0;//nullの代わり
                    }
                }
            }
            double[,,] gradient = new double[2, height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // gradient in the x direction
                    if (x == 0)
                    {
                        gradient[0, y, x] = grey_image[y, x + 1] - grey_image[y, x];
                    }
                    else if (x == width - 1)
                    {
                        gradient[0, y, x] = grey_image[y, x] - grey_image[y, x - 1];
                    }
                    else
                    {
                        gradient[0, y, x] = (grey_image[y, x + 1] - grey_image[y, x - 1]) / 2.0;
                    }

                    // gradient in the y direction
                    if (y == 0)
                    {
                        gradient[1, y, x] = grey_image[y + 1, x] - grey_image[y, x];
                    }
                    else if (y == height - 1)
                    {
                        gradient[1, y, x] = grey_image[y, x] - grey_image[y - 1, x];
                    }
                    else
                    {
                        gradient[1, y, x] = (grey_image[y + 1, x] - grey_image[y - 1, x]) / 2.0;
                    }
                }
            }

            double[,] gradient_val = new double[height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    gradient_val[y, x] = Math.Sqrt(Math.Pow(gradient[0, y, x], 2) + Math.Pow(gradient[1, y, x], 2));
                }
            }

            double[,,] max_gradient = new double[height, width, 2];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (this.front[y, x] == 1)
                    {
                        var patch = this._get_patch(new int[] { y, x });
                        var patchYGradient = _patch_data2D(Get2DArrayFrom3DArray(gradient, 0), patch);//3次元配列から2次元配列を取り出す
                        var patchXGradient = _patch_data2D(Get2DArrayFrom3DArray(gradient, 1), patch);
                        var patchGradientVal = _patch_data2D(gradient_val, patch);

                        double maxGradientVal = double.MinValue;
                        int maxX = 0;
                        int maxY = 0;
                        for (int patchY = 0; patchY < patchGradientVal.GetLength(0); patchY++)
                        {
                            for (int patchX = 0; patchX < patchGradientVal.GetLength(1); patchX++)
                            {
                                if (patchGradientVal[patchY, patchX] > maxGradientVal)
                                {
                                    maxGradientVal = patchGradientVal[patchY, patchX];
                                    maxY = patchY;
                                    maxX = patchX;
                                }
                            }
                        }

                        max_gradient[y, x, 0] = patchYGradient[maxY, maxX];
                        max_gradient[y, x, 1] = patchXGradient[maxY, maxX];
                    }
                }
            }
            return max_gradient;
        }

        public static double[,] Get2DArrayFrom3DArray(double[,,] source, int num)//3次元の配列（RGB画像）から欲しい２次元配列のみを取得する
        {
            int height = source.GetLength(1);
            int width = source.GetLength(2);
            var result = new double[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    result[i, j] = source[num, i, j];
                }
            }
            return result;
        }


        public static uint[,] rgb2gray(uint[,,] source)
        {
            // グレースケール画像を生成
            int w = source.GetLength(0);
            int h = source.GetLength(1);
            uint[,] data = new uint[w, h];

            // int[,]のpixel値を配列に挿入
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    // グレイスケール変換処理
                    data[j, i] = (source[j, i, 0] + source[j, i, 1] + source[j, i, 2]) / 3;
                }
            }
            return data;
        }


        public (int maxX, int maxY) _find_highest_priority_pixel() {
            {
                int maxX = 0;
                int maxY = 0;
                double maxPriority = double.MinValue;

                int height = this.priority.GetLength(0);
                int width = this.priority.GetLength(1);
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        if (this.priority[y, x] > maxPriority)
                        {
                            maxPriority = this.priority[y, x];
                            maxY = y;
                            maxX = x;
                        }
                    }
                }
                UnityEngine.Debug.Log(string.Format("Highest Priority: [{0},{1}]", maxX, maxY));
                return (maxX, maxY);
            }
        }


        List<int[]> _find_source_patch(int[] target_pixel){
                // 優先度が高いピクセルの周りのパッチを取得する
                var target_patch = this._get_patch(target_pixel);
                // 画像の高さと幅を取得する
                int height = this.working_image.GetLength(0);
                int width = this.working_image.GetLength(1);
                // モデル画像の高さと幅を取得する
                int model_height = this.model_image.GetLength(0);
                int model_width = this.model_image.GetLength(1);
                // パッチの高さと幅を取得する
                int patch_height, patch_width;
                _patch_shape(target_patch, out patch_height, out patch_width);
            // 一致するパッチを保存する変数を用意する
            List<int[]> best_match = null;
                // 一致するパッチとの最小の差異を保存する変数を用意する
                double best_match_difference = 0;
                // 画像をLab色空間に変換する
                var lab_image = rgb2lab(this.working_image);
                var lab_model_image = rgb2lab(this.model_image);
                // source_patchをランダムに決める 
                System.Random rnd = new System.Random();
                for (int pnum = 0; pnum < height * width / 100; pnum++)//用検討
                {
                    // モデル画像内でランダムな座標を選択する
                    int y = rnd.Next(0, model_height - patch_height);
                    int x = rnd.Next(0, model_width - patch_width);
                    // 選択した座標からパッチを取得する
                    var source_patch = new List<int[]>
                {
                    new int[] {
                        y,
                        y + patch_height - 1
                    },
                    new int[] {
                        x,
                        x + patch_width - 1
                    }
                };
                    // モデル画像の範囲を超えていた場合は処理を終了する
                    if (source_patch[0][1] >= model_height || source_patch[1][1] >= model_width)
                    {
                        break;
                    }
                    // target_patchとsource_patchの差異を計算する
                    var difference = this._calc_patch_difference(lab_image, lab_model_image, target_patch, source_patch);
                    // 差異が最小の場合、best_matchを更新する
                    if (best_match == null || difference < best_match_difference)
                    {
                        best_match = source_patch;
                        best_match_difference = difference;
                    }
                    // 差異が0の場合は、最良のパッチが見つかったので処理を終了する
                    if (difference == 0)
                    {
                        return best_match;
                    }
                }
                // 最良のパッチを返す
                /*for y in range(height - patch_height + 1):#作業画像の範囲←ここをランダムにする
                for x in range(width - patch_width + 1):
                    source_patch = [
                        [y, y + patch_height-1],
                        [x, x + patch_width-1]
                    ]

                    if source_patch[0][1] >= model_height or source_patch[1][1] >= model_width:
                        break

                    difference = self._calc_patch_difference(
                        lab_image,
                        lab_model_image,median
                        target_patch,
                        source_patch
                    )

                    if best_match is None or difference < best_match_difference:
                        best_match = source_patch
                        best_match_difference = difference

                    if difference == 0:
                        return best_match*/
          return best_match;
        }

        public static double[,,] rgb2lab(uint[,,] image)
        {
            double[,,] labImage = new double[image.GetLength(0), image.GetLength(1), 3];
            for (int x = 0; x < image.GetLength(1); x++)
            {
                for (int y = 0; y < image.GetLength(0); y++)
                {
                    double r = image[y, x, 0] / 255.0;
                    double g = image[y, x, 1] / 255.0;
                    double b = image[y, x, 2] / 255.0;

                    // RGBからXYZに変換
                    double X = 0.412453 * r + 0.357580 * g + 0.180423 * b;
                    double Y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
                    double Z = 0.019334 * r + 0.119193 * g + 0.950227 * b;

                    // XYZからLabに変換
                    double epsilon = 0.008856;  // 白色点の補正用のパラメータ
                    double kappa = 903.3;       // 白色点の補正用のパラメータ
                    double Xr = 0.950456;       // 参照白色点
                    double Yr = 1.0;            // 参照白色点
                    double Zr = 1.088754;       // 参照白色点
                    double fx = X / Xr;
                    double fy = Y / Yr;
                    double fz = Z / Zr;
                    double L = (fy > epsilon) ? 116 * Math.Pow(fy, 1.0 / 3.0) - 16 : kappa * fy;
                    double A = 500 * (fx - fy);
                    double B = 200 * (fy - fz);

                    // Labに変換した値を画像に適用
                    labImage[y, x, 0] = L; //(x, y, Color.FromArgb(255, (int)L, (int)a, (int)b));
                    labImage[y, x, 1] = A;
                    labImage[y, x, 2] = B;
                }
            }
            return labImage;
        }



        public double _calc_patch_difference(double[,,] image, double[,,] model_image, List<int[]> target_patch, List<int[]> source_patch)
        {
            double[,,] target_data = _patch_data(image, target_patch);
            double[,,] source_data = _patch_data(model_image, source_patch);
            //double absolute_distance = Math.Abs(target_data - source_data);
            double squared_distance = 0;
            for (int x = 0; x < target_data.GetLength(1); x++)
            {
                for (int y = 0; y < target_data.GetLength(0); y++)
                {
                    for (int k = 0; k < target_data.GetLength(2); k++)
                    {
                        squared_distance += Math.Pow(target_data[y, x, k] - source_data[y, x, k], 2);

                    }
                }
            }
          //double euclidean_distance = Math.Sqrt(Math.Pow(target_patch[0][0] - source_patch[0][0], 2) + Math.Pow(target_patch[1][0] - source_patch[1][0], 2));
          return squared_distance;
        }

        public void _update_image(int[] target_pixel, List<int[]> source_patch) {
            var target_patch = this._get_patch(target_pixel);
            var pixels_positions = GetPixelsPositions(target_patch);
            var patch_confidence = this.confidence[target_pixel[0], target_pixel[1]];
            for (int i = 0; i < pixels_positions.Count; i++)
            {
                int[] point = pixels_positions[i];
                this.confidence[point[0], point[1]] = patch_confidence;
            }
            //var mask = _patch_data2Duint(this.working_mask, target_patch);
            // var rgb_mask = this._to_rgb(mask);
            var source_data = _patch_datauint(this.model_depth, source_patch);
            //var target_data = this._patch_data(this.working_image, target_patch);
            //new_data = source_data * rgb_mask + target_data * (1 - rgb_mask)
            var new_data = source_data;
            _copy_to_patch(this.output_image, target_patch, new_data);
            _copy_to_patch2D(this.working_mask, target_patch);//0代入するだけ
        }


        private List<int[]> GetPixelsPositions(List<int[]> target_patch)//パッチの座標取得
        {
            List<int[]> pixels_positions = new List<int[]>();
            for (int i = target_patch[0][0]; i <= target_patch[0][1]; i++)
            {
                for (int j = target_patch[1][0]; j <= target_patch[1][1]; j++)
                {
                    if (this.working_mask[i, j] == 1)
                    {
                        pixels_positions.Add(new int[] { i, j });
                    }
                }
            }
            return pixels_positions;
        }

        public virtual bool _finished() {
            var height = working_image.GetLength(0);
            var width = working_image.GetLength(1);
            var remaining = SumWorkingMask();
            var total = height * width;

            UnityEngine.Debug.Log(string.Format("{0} of {1} completed", total - remaining, total));
            return remaining == 0;
        }

        private uint SumWorkingMask()//推定完了していない部分の合計を取得
        {
            uint sum = 0;
            for (int i = 0; i < this.working_mask.GetLength(0); i++)
            {
                for (int j = 0; j < this.working_mask.GetLength(1); j++)
                {
                    sum += this.working_mask[i, j];
                }
            }
            return sum;
        }


        public static void _patch_shape(List<int[]> patch, out int height, out int width)
        {
            height = 1 + patch[0][1] - patch[0][0];
            width = 1 + patch[1][1] - patch[1][0];
        }

        public static double[,,] _patch_data(double[,,] source, List<int[]> patch)//patchには座標,sourceには中身を
        {//パッチの範囲から取得したデータを新しいリストに
            int patchHeight = patch[0][1] - patch[0][0] + 1;
            int patchWidth = patch[1][1] - patch[1][0] + 1;
            int patchChannels = source.GetLength(2);
            var patchData = new double[patchHeight, patchWidth, patchChannels];

            for (int i = patch[0][0]; i <= patch[0][1]; i++)
            {
                for (int j = patch[1][0]; j <= patch[1][1]; j++)
                {
                    for (int k = 0; k < patchChannels; k++)
                    {
                        patchData[i - patch[0][0], j - patch[1][0], k] = source[i, j, k];
                    }
                }
            }
            return patchData;
        }
        public static uint[,,] _patch_datauint(uint[,,] source, List<int[]> patch)//patchには座標,sourceには中身を
        {//パッチの範囲から取得したデータを新しいリストに
            int patchHeight = patch[0][1] - patch[0][0] + 1;
            int patchWidth = patch[1][1] - patch[1][0] + 1;
            int patchChannels = source.GetLength(2);
            var patchData = new uint[patchHeight, patchWidth, patchChannels];

            for (int i = patch[0][0]; i <= patch[0][1]; i++)
            {
                for (int j = patch[1][0]; j <= patch[1][1]; j++)
                {
                    for (int k = 0; k < patchChannels; k++)
                    {
                        patchData[i - patch[0][0], j - patch[1][0], k] = source[i, j, k];
                    }
                }
            }
            return patchData;
        }

        public static double[,] _patch_data2D(double[,] source, List<int[]> patch)//patchには座標,sourceには中身を
        {//パッチの範囲から取得したデータを新しいリストに
            //UnityEngine.Debug.Log(string.Format("patch[{0},{1}]", source.GetLength(0), source.GetLength(1)));
            var patchData = new double[patch[0][1] - patch[0][0] + 1, patch[1][1] - patch[1][0] + 1];
            UnityEngine.Debug.Log(string.Format("patch[{0},{1}],[{2},{3}]", patch[0][0], patch[0][1], patch[1][0], patch[1][1]));
            for (int i = patch[0][0]; i <= patch[0][1]; i++)
            {
                for (int j = patch[1][0]; j <= patch[1][1]; j++)
                {
                    patchData[i - patch[0][0], j - patch[1][0]] = source[i, j];
                }
            }
            return patchData;
        }

        public static uint[,] _patch_data2Duint(uint[,] source, List<int[]> patch)//patchには座標,sourceには中身を
        {//パッチの範囲から取得したデータを新しいリストに
            var patchData = new uint[patch[0][1] - patch[0][0] + 1, patch[1][1] - patch[1][0] + 1];
            for (int i = patch[0][0]; i <= patch[0][1]; i++)
            {
                for (int j = patch[1][0]; j <= patch[1][1]; j++)
                {
                    patchData[i - patch[0][0], j - patch[1][0]] = source[i, j];
                }
            }
            return patchData;
        }




        public static uint[,,] _to_rgb(uint[,] image)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);

            // RGB画像を生成
            uint[,,] rgbImage = new uint[height, width, 3];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        rgbImage[y, x, k] = image[y, x];
                    }
                }
            }
            return rgbImage;
        }


        public static void _copy_to_patch(uint[,,] dest, List<int[]> dest_patch, uint[,,] data)
        {
            int dest_height = dest_patch[0][1] - dest_patch[0][0] + 1;
            int dest_width = dest_patch[1][1] - dest_patch[1][0] + 1;
            for (int i = 0; i < dest_height; i++)
            {
                for (int j = 0; j < dest_width; j++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        dest[i + dest_patch[0][0], j + dest_patch[1][0], k] = data[i, j, k];
                    }
                }
            }
        }

        public static void _copy_to_patch2D(uint[,] dest, List<int[]> dest_patch)//0を代入する目的
        {
            int dest_height = dest_patch[0][1] - dest_patch[0][0] + 1;
            int dest_width = dest_patch[1][1] - dest_patch[1][0] + 1;
            for (int i = 0; i < dest_height; i++)
            {
                for (int j = 0; j < dest_width; j++)
                {
                    // set the pixel color of dest Texture2D
                    dest[i + dest_patch[0][0], j + dest_patch[1][0]] = 0;
                }
            }
        }

        public static uint[,] Texture2DToUintArray(Texture2D source)
        {
            int width = source.width;
            int height = source.height;
            Color32[] pixels = source.GetPixels32();
            uint[,] data = new uint[height, width];

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    Color32 pixel = pixels[i * width + j];
                    data[i, j] = (uint)((pixel.r + pixel.g + pixel.b) / 3);
                }
            }
            return data;
        }

        public static uint[,,] Texture2DToUintArray3D(Texture2D source)//3次元にすることでカラーの情報が得られる
        {
            int width = source.width;
            int height = source.height;
            Color32[] pixels = source.GetPixels32();
            uint[,,] data = new uint[height, width, 3];

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    Color32 pixel = pixels[i * width + j];
                    data[i, j, 0] = pixel.r;
                    data[i, j, 1] = pixel.g;
                    data[i, j, 2] = pixel.b;
                }
            }
            return data;
        }


    }
    
    public class Median {
        
        public Texture2D image;
        
        public Median(Texture2D image) {
            this.image = image;
        }

        public Texture2D median()
        {
            // Convert to grayscale
            Texture2D gray = Grayscale(image);

            // Apply median filter
            var median = MedianBlur(gray, 3);

            // Convert back to RGB
            Texture2D dst = ConvertToRGB(median);

            return dst;
        }

        // Convert to grayscale
        private static Texture2D Grayscale(Texture2D bmp)
        {
            Texture2D ret = new Texture2D(bmp.width, bmp.height);
            for (int y = 0; y < bmp.height; y++)
                for (int x = 0; x < bmp.width; x++)
                {
                    var px = bmp.GetPixel(x, y);
                    var gray = (int)(px.r * 0.3 + px.g * 0.59 + px.b * 0.11);
                    ret.SetPixel(x, y, new UnityEngine.Color(gray, gray, gray));
                }
            return ret;
        }

        private static Texture2D MedianBlur(Texture2D bmp, int kernelSize)
        {
            var ret = new Texture2D(bmp.width, bmp.height);
            int kernSize = kernelSize;

            for (int x = kernSize / 2; x < ret.width - kernSize / 2; x++)
                for (int y = kernSize / 2; y < ret.height - kernSize / 2; y++)
                {
                    var window = new List<int>();
                    for (int i = -kernSize / 2; i <= kernSize / 2; i++)
                        for (int j = -kernSize / 2; j <= kernSize / 2; j++)
                        {
                            int newX = x + i;
                            int newY = y + j;

                            window.Add((int)bmp.GetPixel(newX, newY).r);
                        }

                    var colorList = new int[window.Count];
                    window.CopyTo(colorList);
                    Array.Sort(colorList);

                    ret.SetPixel(x, y, new UnityEngine.Color(colorList[(kernSize * kernSize) / 2], colorList[(kernSize * kernSize) / 2], colorList[(kernSize * kernSize) / 2]));
                }
            return ret;
        }

        // Convert back to RGB
        private static Texture2D ConvertToRGB(Texture2D bmp)
        {
            var ret = new Texture2D(bmp.width, bmp.height);
            for (int y = 0; y < bmp.height; y++)
                for (int x = 0; x < bmp.width; x++)
                {
                    var px = bmp.GetPixel(x, y);
                    var gray = px.r;
                    ret.SetPixel(x, y, new UnityEngine.Color(gray, gray, gray));
                }
            return ret;
        }


    }
    
}
