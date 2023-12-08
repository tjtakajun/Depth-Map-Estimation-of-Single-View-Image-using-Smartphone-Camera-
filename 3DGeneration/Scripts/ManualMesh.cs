using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

public class ManualMesh : MonoBehaviour
{
    public Texture2D depthImage; // 手動で準備した白黒画像をInspectorからアタッチする

    private Mesh _mesh;
    public RawImage depth;
    private GameObject ThreeDRecon;
    private List<Vector3> _vertices = new List<Vector3>();
    private List<Color32> _colors = new List<Color32>();
    private List<int> _indices = new List<int>();

    // private void Start()
    //{
    //     DepthMap = GameObject.Find("DepthMap");
    // }

    public GameObject InitializePointCloud()
    {
        // メッシュを作成
        _mesh = new Mesh();
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

        // 画像をテクスチャとしてマテリアルに設定

        // メッシュの生成


        // 点群のメッシュを作成してAR空間に配置
        GameObject pointCloudObject = new GameObject("PointCloud");
        MeshFilter meshFilter = pointCloudObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = pointCloudObject.AddComponent<MeshRenderer>();
        meshFilter.mesh = _mesh;


        // ポイントクラウドオブジェクトの位置をAR空間に合わせる
        pointCloudObject.transform.position = transform.position;
        pointCloudObject.transform.rotation = transform.rotation;

        return pointCloudObject; // 生成した点群オブジェクトを返す
    }

    public void PickDepthMap()
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
                depthImage = createReadabeTexture2D(texture);
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



    void MeshGen()
    {
        

    }

    // 輝度値を補間して取得するメソッド
    private float InterpolateBrightness(Color32[] pixels, int x, int y, int width, int height, float minBrightness, float maxBrightness)
    {
        // 画素座標が画像範囲外の場合、補間しないで輝度値の最小値を返す
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            return minBrightness;
        }

        int index = y * width + x;
        Color32 pixel = pixels[index];

        // RGB値の平均を輝度値として取得
        float brightness = (pixel.r + pixel.g + pixel.b) / 3f;

        // 輝度値を0から1に正規化
        float normalizedBrightness = Mathf.InverseLerp(minBrightness, maxBrightness, brightness);

        // 厚みの倍率を設定（1より大きい値を設定すると厚みが大きくなります）
        float thicknessMultiplier = 3f;

        // 正規化された輝度値に倍率をかける
        normalizedBrightness *= thicknessMultiplier;

        return normalizedBrightness;
    }
}
