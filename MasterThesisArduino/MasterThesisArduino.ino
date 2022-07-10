/* How to use the DHT-22 sensor with Arduino uno
   Temperature and humidity sensor
   More info: http://www.ardumotive.com/how-to-use-dht-22-sensor-en.html
   Dev: Michalis Vasilakis // Date: 1/7/2015 // www.ardumotive.com */

//Libraries
#include <DHT.h>

//Constants
#define DHTPIN 2     // what pin we're connected to
#define DHTTYPE DHT22   // DHT 22  (AM2302)
DHT dht(DHTPIN, DHTTYPE); //// Initialize DHT sensor for normal 16mhz Arduino


//Variables
int chk;
float hum;  //Stores humidity value
float temp; //Stores temperature value
float light;

void setup()
{
    Serial.begin(9600);
  dht.begin();

}

void loop()
{
     
    //Read data and store it to variables hum and temp
    hum = dht.readHumidity();
    temp= dht.readTemperature();
    light = analogRead(A0);

    hum = map(hum, 0, 100, 0, 1000);
    temp = map(temp, -40, 80, 0, 1000);
    light = map(light, 0, 1023, 0, 1000);


    Serial.print(light);   // the raw analog reading
    Serial.print(", ");
    Serial.print(hum);   // the raw analog reading
    Serial.print(", ");
    Serial.println(temp);   // the raw analog reading  
    delay(1000); //Delay 2 sec.
}
   
