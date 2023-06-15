using System.Collections;
using System.Collections.Generic;
using UnityEngine;

    public class OnReceiveData : MonoBehaviour
    {
        // Captureシーンから受け取ったデータを使用する
        public void OnReceive(Texture2D yourImage)
        {
            // Image Cropper Prefabを取得します
            //ImageCropper imageCropper = GetComponent<ImageCropper>();

            // Texture2DからSpriteを作成し、Imageコンポーネントに設定する
            //var sprite = Sprite.Create(yourImage, new Rect(0, 0, yourImage.width, yourImage.height), Vector2.zero);
            //Texture texture = sprite.texture;

            //ImageCropper.Instance.OrientedImage = yourImage;

            // 画像を表示します
            ImageCropperRE.CropResult onCrop = (bool result, Texture originalImage, Texture2D croppedImage) =>
            {
                // 切り取り作業が成功した場合
                if (result)
                {
                    ImageCropperRE.Instance.Crop();
                    // 切り取った画像を使用します
                    // ここでは例として画像を表示するImageコンポーネントに設定します

                }
                else
                {
                    //ImageCropper.Instance.Cancel();
                    // 切り取り作業がキャンセルされた場合の処理を記述します
                }
            };

            // 画像を表示します
            ImageCropperRE.Instance.Show(yourImage, onCrop, null, null);

        }
    }

