using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{

    int width = 320;
    int height = 180;
    int fps = 60;
    WebCamTexture webcamTexture;

    public void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        webcamTexture = new WebCamTexture(devices[0].name, this.width, this.height, this.fps);
        GetComponent<Renderer>().material.mainTexture = webcamTexture;
        webcamTexture.Play();
    }
}
