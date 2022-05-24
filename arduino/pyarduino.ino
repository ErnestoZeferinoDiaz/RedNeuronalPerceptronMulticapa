
int flex0=A0;
int flex1=A1;
int flex2=A2;
int flex3=A3;
int flex4=A4;

float r0,r1,r2,r3,r4;
String tmp;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(flex0,INPUT);
  pinMode(flex1,INPUT);
  pinMode(flex2,INPUT);
  pinMode(flex3,INPUT);
  pinMode(flex4,INPUT);
  
}

void loop() {
  // put your main code here, to run repeatedly:
  r0 = analogRead(flex0);
  r1 = analogRead(flex1);
  r2 = analogRead(flex2);
  r3 = analogRead(flex3);
  r4 = analogRead(flex4);

  tmp="";
  tmp.concat(r0);
  tmp.concat(",");
  tmp.concat(r1);
  tmp.concat(",");
  tmp.concat(r2);
  tmp.concat(",");
  tmp.concat(r3);
  tmp.concat(",");
  tmp.concat(r4);
  tmp.concat(",");
  tmp.concat(26);
  Serial.println(tmp);
}
