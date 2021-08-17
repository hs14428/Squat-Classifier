package com.company;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public class API_POST {

    private String TOKEN = "3de90533da13f409953870b54506da46df848224";

    public static void main(String[] args) throws IOException {
        // write your code here
        String FILEPATH = "files/path/home/hs14428/mysite";
        String API = "https://www.pythonanywhere.com/api/v0/user/hs14428/";
        String FILENAME = "/test.mp4";
        String FULL_URL = API + FILEPATH + FILENAME;
        System.out.println(FULL_URL);
        new API_POST().sendRequest(FULL_URL);
    }

    void sendRequest(String request) throws IOException
    {
        URL url = new URL(request);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setDoOutput(true);
        connection.setInstanceFollowRedirects(false);
        connection.setRequestMethod("POST");
        connection.setRequestProperty("Content-Type", "video/mp4");
        connection.setRequestProperty("charset", "utf-8");
        connection.setRequestProperty("Authorization", "Token "+this.TOKEN);

        File file = new File("test.mp4");
        byte[] postData = new byte[(int) file.length()];

        FileInputStream fis = new FileInputStream(file);
        fis.read(postData); // Read file into bytes[]
        fis.close();

//        byte[] postData = Files.readAllBytes(file.toPath());
        DataOutputStream wr = new DataOutputStream(connection.getOutputStream());
        wr.write(postData);
        wr.flush();
        wr.close();

        System.out.println(connection.getResponseCode());
    }
}
