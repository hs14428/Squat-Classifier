package com.company;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class API_GET {

    public static void main(String[] args) throws IOException {
	// write your code here
        String BASE_URL = "http://hs14428.pythonanywhere.com/?input=wagwarn gangsta";
        new API_GET().sendRequest(BASE_URL);
    }

    void sendRequest(String request) throws IOException
    {
        URL url = new URL(request);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
//        connection.setDoOutput(true);
//        connection.setInstanceFollowRedirects(false);
        connection.setRequestMethod("GET");
        connection.setRequestProperty("Content-Type", "text/plain");
        connection.setRequestProperty("charset", "utf-8");
        connection.connect();
        System.out.println(connection.getResponseCode());

        BufferedReader in = new BufferedReader(
                new InputStreamReader(connection.getInputStream()));
        String inputLine;
        StringBuffer response = new StringBuffer();

        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();
        System.out.println(response.toString());
    }
}
