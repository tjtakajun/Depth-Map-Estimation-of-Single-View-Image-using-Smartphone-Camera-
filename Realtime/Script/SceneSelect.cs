using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
//using UnityEngine.EventSystems;



public class SceneSelect : MonoBehaviour
{
    public RawImage depth;
    public RawImage image;
    private Texture2D m_Texture;
    // Declare a field to store a reference to the ARCameraManager component.
    private ARCameraManager cameraManager;
    private GameObject demoCarouselUI;
    private GameObject ModelImage;
    private GameObject ModelDepth;
    private GameObject CapButton;

    // In the Awake method, get a reference to the ARCameraManager component.
    private void Awake()
    {
       
    }

    // Start is called before the first frame update
    private void Start()
    {
        CapButton = GameObject.Find("To Realtime");
        ModelImage = GameObject.Find("modelimage");
        ModelDepth = GameObject.Find("modeldepth");
        demoCarouselUI = GameObject.Find("Carousel UI");

        ModelDepth.SetActive(false);
        ModelImage.SetActive(false);
        // Get a reference to the ARFDepthComponents game object.
        GameObject arfDepthComponentsObject = GameObject.Find("ARFDepthComponents1");
        if (arfDepthComponentsObject == null)
        {
            UnityEngine.Debug.LogError("arfDepthComponentsObject not found.");
            return;
        }
        UnityEngine.Debug.Log("ARFDepthComponent found.");

        // Get a reference to the ARSessionOrigin game object under ARFDepthComponents.
        GameObject arSessionOriginObject = arfDepthComponentsObject.transform.Find("AR Session Origin").gameObject;
        if (arSessionOriginObject == null)
        {
            UnityEngine.Debug.LogError("arSessionOriginObject not found.");
            return;
        }
        UnityEngine.Debug.LogError("ARSessionOrigin found.");

        // Get a reference to the ARCamera game object under ARSessionOrigin.
        GameObject arCameraObject = arSessionOriginObject.transform.Find("AR Camera").gameObject;
        if (arCameraObject == null)
        {
            UnityEngine.Debug.LogError("ARCamera game object not found.");
            return;
        }
        else
            UnityEngine.Debug.LogError("ARCamera found.");

        // Get a reference to the ARCamera Manager component of the ARCamera game object.
        cameraManager = arCameraObject.GetComponent<ARCameraManager>();

        if (cameraManager == null)
        {
            UnityEngine.Debug.LogError("ARCameraManager component not found.");
            return;
        }
        UnityEngine.Debug.Log("ARCameraManager component found.");

        // カメラフレームを受け取るたびにOnCameraFrameReceivedメソッドを呼び出す
        cameraManager.frameReceived += OnCameraFrameReceived;
    }

 /*   public void OnPointerClick(PointerEventData eventData)
    {
        UnityEngine.Debug.Log("RawImageがクリックされました");
        if (associatedRawImage.name == "A")
        {
            YourScriptName.YourFunctionNameA(associatedRawImage);
        }
        else if (associatedRawImage.name == "B")
        {
            YourScriptName.YourFunctionNameB(associatedRawImage);
        }
    }*/

    //public Texture2D Image;
    public Texture2D modeldepth;
    public Texture2D modelimage;
    public int patch_size;//12
    public string output;
    public bool plotprogress;
    public void ChangeScene()
    {
        // If image cropper is already open, do nothing
        if (ImageCropper.Instance.IsOpen)
            return;
        ModelDepth.SetActive(true);
        ModelImage.SetActive(true);
        //OnCameraFrameReceivedは，カメラフレームが呼び出されるたび実行
        StartCoroutine(OnReceiveData());
    }

    // Captureシーンから受け取ったデータを使用する
    private IEnumerator OnReceiveData()
    {
        Texture2D tex = new Texture2D(m_Texture.width, m_Texture.height, TextureFormat.RGB24, false);
        tex = m_Texture;
        UnityEngine.Debug.Log("OnReceiveData.");
        yield return new WaitForEndOfFrame();
        

        // 画像を表示します
        ImageCropper.CropResult onCrop = (bool result, Texture originalImage, Texture2D croppedImage) =>
        {
            // 切り取り作業が成功した場合
            if (result)
            {
                if(croppedImage == null)
                {
                    UnityEngine.Debug.LogError("Cropping was failed!");
                }
                else
                {
                    UnityEngine.Debug.Log("Cropping succeeded!");
                }
                Texture2D Crop = createReadabeTexture2D(croppedImage);
                //Texture2D Crop = createReadabeTexture2D(Image);
                byte[] bytes = Crop.EncodeToPNG();
                File.WriteAllBytes("/storage/emulated/0/DCIM/Camera/crop.png", bytes);
                depthpatch Mydepth = new depthpatch();
                Mydepth.main(this.modeldepth, this.modelimage, this.patch_size, this.output, this.plotprogress, Crop);//深度マップ推定
                demoCarouselUI.SetActive(true);
                CapButton.SetActive(true);
                ModelDepth.SetActive(false);
                ModelImage.SetActive(false);
                cameraManager.frameReceived += OnCameraFrameReceived;
            }
            else
            {
                // 切り取り作業がキャンセルされた場合の処理を記述します
                demoCarouselUI.SetActive(true);
                CapButton.SetActive(true);
                ModelDepth.SetActive(false);
                ModelImage.SetActive(false);
                cameraManager.frameReceived += OnCameraFrameReceived;
            }
            tex = null;
            Resources.UnloadUnusedAssets();
        };
        cameraManager.frameReceived -= OnCameraFrameReceived;
        // 画像を表示します
        ImageCropper.Instance.Show(tex, onCrop, null, null);
        ModelDepth.SetActive(true);
        ModelImage.SetActive(true);
        demoCarouselUI.SetActive(false);
        CapButton.SetActive(false);
    }
    public void PickModelImage()
    {
        // 画像の読み込み
        NativeGallery.Permission permission = NativeGallery.GetImageFromGallery((path) =>
        {
            UnityEngine.Debug.Log("Image path: " + path);
            if (path != null)
            {
                // 画像パスからTexture2Dを生成
                Texture2D texture = NativeGallery.LoadImageAtPath(path, 512);
                if (texture == null)
                {
                    UnityEngine.Debug.Log("Couldn't load texture from " + path);
                    return;
                }
                // RawImageの元テクスチャを破棄
                Destroy(image.texture);

                // RawImageで新規テクスチャを表示
                image.texture = texture;
                modelimage = createReadabeTexture2D(texture);
            }
        });
        UnityEngine.Debug.Log("Permission result: " + permission);
    }

    public void PickModelDepth()
    {
        // 画像の読み込み
        NativeGallery.Permission permission = NativeGallery.GetImageFromGallery((path) =>
        {
            UnityEngine.Debug.Log("Image path: " + path);
            if (path != null)
            {
                // 画像パスからTexture2Dを生成
                Texture2D texture = NativeGallery.LoadImageAtPath(path, 512);
                if (texture == null)
                {
                    UnityEngine.Debug.Log("Couldn't load texture from " + path);
                    return;
                }

                // RawImageの元テクスチャを破棄
                Destroy(depth.texture);

                // RawImageで新規テクスチャを表示
                depth.texture = texture;
                modeldepth = createReadabeTexture2D(texture);
            }
        });
        UnityEngine.Debug.Log("Permission result: " + permission);
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

    unsafe void OnCameraFrameReceived(ARCameraFrameEventArgs eventArgs)
    {
        UnityEngine.Debug.Log("OnCameraFrameReceived called.");

        if (!cameraManager.TryAcquireLatestCpuImage(out XRCpuImage image))
        {
            UnityEngine.Debug.Log("Failed to acquire latest image from camera manager.");
            return;
        }
        else
        {
            UnityEngine.Debug.Log("Acquire latest image from camera manager.");
        }

        var conversionParams = new XRCpuImage.ConversionParams
        {
            // Get the entire image.
            inputRect = new RectInt(0, 0, image.width, image.height),

            // Downsample by 2.
            outputDimensions = new Vector2Int(image.width, image.height),

            // Choose RGBA format.
            outputFormat = TextureFormat.RGBA32,

            // Flip across the vertical axis (mirror image).
            transformation = XRCpuImage.Transformation.MirrorY
        };

        // See how many bytes you need to store the final image.
        int size = image.GetConvertedDataSize(conversionParams);

        // Allocate a buffer to store the image.
        var buffer = new NativeArray<byte>(size, Allocator.Temp);

        // Extract the image data
        image.Convert(conversionParams, new IntPtr(buffer.GetUnsafePtr()), buffer.Length);

        // The image was converted to RGBA32 format and written into the provided buffer
        // so you can dispose of the XRCpuImage. You must do this or it will leak resources.
        image.Dispose();

        // At this point, you can process the image, pass it to a computer vision algorithm, etc.
        // In this example, you apply it to a texture to visualize it.

        // You've got the data; let's put it into a texture so you can visualize it.
        m_Texture = new Texture2D(
            conversionParams.outputDimensions.x,
            conversionParams.outputDimensions.y,
            conversionParams.outputFormat,
            false
        );

        m_Texture.LoadRawTextureData(buffer);
        m_Texture.Apply();

        // Done with your temporary data, so you can dispose it.
        buffer.Dispose();
    }

}