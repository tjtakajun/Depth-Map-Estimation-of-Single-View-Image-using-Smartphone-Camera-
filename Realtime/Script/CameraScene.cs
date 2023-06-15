using System.Collections;
using System.Collections.Generic;
using UnityEngine.SceneManagement;
using UnityEngine;

public class CaptureScene : MonoBehaviour
{
    // カメラからの画像を取得するためのWebCamTexture
    private WebCamTexture webCamTexture;

    public void CameraCapture()
    {
        // WebCamTextureを作成してカメラからの画像を取得を開始する
        webCamTexture = new WebCamTexture();
        webCamTexture.Play();

        // 画像をTexture2Dに変換する
        var texture2D = new Texture2D(webCamTexture.width, webCamTexture.height);
        texture2D.SetPixels(webCamTexture.GetPixels());

        // Realtimeシーンにデータを渡す
        SceneManager.SetActiveScene(SceneManager.GetSceneByName("Realtime"));
        var scene = SceneManager.GetSceneByName("Realtime");
        var rootGameObjects = scene.GetRootGameObjects();
        foreach (var gameObject in rootGameObjects)
        {
            gameObject.SendMessage("OnReceiveData", texture2D);
        }
    }

    // Captureシーンから受け取ったデータを使用する
    public void OnReceiveData(Texture2D yourImage)
    {
        // Image Cropper Prefabを取得します
        ImageCropper imageCropper = GetComponent<ImageCropper>();

        // 画像を表示します
        ImageCropper.CropResult onCrop = (bool result, Texture originalImage, Texture2D croppedImage) =>
        {
            // 切り取り作業が成功した場合
            if (result)
            {
                // 切り取った画像を使用します
                // ここでは例として画像を表示するImageコンポーネントに設定します

            }
            else
            {
                // 切り取り作業がキャンセルされた場合の処理を記述します
            }
        };

        // 画像を表示します
        ImageCropper.Instance.Show(yourImage, onCrop, null, null);

    }
}
