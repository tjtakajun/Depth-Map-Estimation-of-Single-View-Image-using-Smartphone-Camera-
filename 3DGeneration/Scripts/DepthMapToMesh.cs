using UnityEngine;

public class DepthMapToMesh : MonoBehaviour
{
    public Texture2D depthMap; // �蓮�Ŏw�肷��[�x�摜
    public Material meshMaterial; // ���b�V���ɓK�p����}�e���A��

    private MeshFilter _meshFilter;
    private MeshRenderer _meshRenderer;
    private Mesh _mesh;

    private void Start()
    {
        _meshFilter = GetComponent<MeshFilter>();
        _meshRenderer = GetComponent<MeshRenderer>();
        _mesh = new Mesh();
        _mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

        // �[�x�摜���烁�b�V�����쐬
        SetMeshFromDepthMap();

        // ���b�V���Ƀ}�e���A����K�p
        _meshRenderer.material = meshMaterial;

        // ���b�V����ݒ�
        _meshFilter.mesh = _mesh;
    }

    private void SetMeshFromDepthMap()
    {
        if (depthMap != null)
        {
            int width = depthMap.width;
            int height = depthMap.height;

            // ���_���X�g�ƃC���f�b�N�X���X�g�̏�����
            Vector3[] vertices = new Vector3[width * height];
            int[] indices = new int[width * height];

            int index = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // �[�x�摜���璸�_���W���v�Z
                    Color pixelColor = depthMap.GetPixel(x, y);
                    float depthValue = pixelColor.r; // �[�x�l�͐ԃ`�����l���ɕۑ�����Ă�����̂Ɖ���

                    Vector3 vertex = TransformVertexToWorldSpace(ComputeVertex(x, y, depthValue));

                    // ���_���W�ƃC���f�b�N�X�����X�g�ɒǉ�
                    vertices[index] = vertex;
                    indices[index] = index;
                    index++;
                }
            }

            // ���b�V���ɒ��_���W�ƃC���f�b�N�X��ݒ�
            _mesh.vertices = vertices;
            _mesh.SetIndices(indices, MeshTopology.Points, 0);
            _mesh.RecalculateBounds();
        }
    }

    // �ȉ��̊֐���BackgroundToDepthMapEffectController�N���X����̃R�s�[�ł�

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
