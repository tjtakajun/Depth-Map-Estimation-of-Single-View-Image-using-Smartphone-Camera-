using UnityEngine;

public class DepthMapToMesh : MonoBehaviour
{
    public Texture2D depthMap; // 手動で指定する深度画像
    public Material meshMaterial; // メッシュに適用するマテリアル

    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Mesh _mesh;

    private void Start()
    {
        _meshFilter = GetComponent<MeshFilter>();
        _meshRenderer = GetComponent<MeshRenderer>();
        _mesh = new Mesh();
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

        // 深度画像からメッシュを作成
        SetMeshFromDepthMap();

        // メッシュにマテリアルを適用
        _meshRenderer.material = meshMaterial;

        // メッシュを設定
        _meshFilter.mesh = _mesh;
    }

    private void SetMeshFromDepthMap()
    {
        if (depthMap != null)
        {
            int width = depthMap.width;
            int height = depthMap.height;

            // 頂点リストとインデックスリストの初期化
            Vector3[] vertices = new Vector3[width * height];
            int[] indices = new int[width * height];

            int index = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // 深度画像から頂点座標を計算
                    Color pixelColor = depthMap.GetPixel(x, y);
                    float depthValue = pixelColor.r; // 深度値は赤チャンネルに保存されているものと仮定

                    Vector3 vertex = TransformVertexToWorldSpace(ComputeVertex(x, y, depthValue));

                    // 頂点座標とインデックスをリストに追加
                    vertices[index] = vertex;
                    indices[index] = index;
                    index++;
                }
            }

            // メッシュに頂点座標とインデックスを設定
            _mesh.vertices = vertices;
            _mesh.SetIndices(indices, MeshTopology.Points, 0);
            _mesh.RecalculateBounds();
        }
    }

    // 以下の関数はBackgroundToDepthMapEffectControllerクラスからのコピーです

    private Vector3 TransformVertexToWorldSpace(Vector3 vertex)
    {
        if (Camera.main != null)
        {
            Matrix4x4 worldToLocalMatrix = Camera.main.transform.worldToLocalMatrix;
            return worldToLocalMatrix.MultiplyPoint3x4(vertex);
        }
        else
        {
            return vertex;
        }
    }

    private Vector3 ComputeVertex(int x, int y, float depth)
    {
        Vector3 vertex = new Vector3(x, y, depth);
        return Camera.main.ScreenToWorldPoint(vertex);
    }
}
