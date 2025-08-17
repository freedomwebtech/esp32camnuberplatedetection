#include <esp32cam.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

// WiFi Credentials
const char* WIFI_SSID = "apex";
const char* WIFI_PASS = "freedomtech";

// Web server running on port 80
WebServer server(80);

// Stream handler
void handleStream() {
    WiFiClient client = server.client();
    const char* boundary = "boundarydonotcross";
    String response = 
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=" + String(boundary) + "\r\n\r\n";
    client.print(response);

    while (client.connected()) {
        auto frame = esp32cam::capture();
        if (frame == nullptr) {
            Serial.println("[-] Camera capture failed");
            break;
        }

        client.printf("--%s\r\n", boundary);
        client.println("Content-Type: image/jpeg");
        client.printf("Content-Length: %d\r\n\r\n", frame->size());
        client.write(frame->data(), frame->size());
        client.println();
        delay(50);  // ~20 FPS
    }
}

// Setup
void setup() {
    Serial.begin(115200);
    Serial.println("\n[+] Starting ESP32-CAM...");

    WiFi.begin(WIFI_SSID, WIFI_PASS);
    if (WiFi.waitForConnectResult() != WL_CONNECTED) {
        Serial.println("[-] WiFi Failed!");
        delay(5000);
        ESP.restart();
    }
    Serial.println("[+] WiFi Connected: " + WiFi.localIP().toString());

    // Camera config
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(Resolution::find(800, 600));
    cfg.setJpeg(80);

    if (!Camera.begin(cfg)) {
        Serial.println("[-] Camera Failed!");
        delay(5000);
        ESP.restart();
    }
    Serial.println("[+] Camera Started");

    // Route for streaming
    server.on("/stream", HTTP_GET, handleStream);

    server.begin();
    Serial.println("[+] Server started. Stream at /stream");
    Serial.println("🔗 Open: http://" + WiFi.localIP().toString() + "/stream");
}

// Loop
void loop() {
    server.handleClient();
}