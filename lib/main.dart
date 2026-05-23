import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'theme/app_theme.dart';
import 'router.dart';
import 'screens/splash_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp, DeviceOrientation.portraitDown,
  ]);
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
  ));
  runApp(const YellowSenseApp());
}

class YellowSenseApp extends StatefulWidget {
  const YellowSenseApp({super.key});
  @override
  State<YellowSenseApp> createState() => _YellowSenseAppState();
}

class _YellowSenseAppState extends State<YellowSenseApp> {
  bool _showSplash = true;

  @override
  Widget build(BuildContext context) {
    if (_showSplash) {
      return MaterialApp(
        title: 'YellowSense UIDAI',
        debugShowCheckedModeBanner: false,
        theme: YS.theme,
        home: SplashScreen(onDone: () => setState(() => _showSplash = false)),
      );
    }
    return MaterialApp.router(
      title: 'YellowSense UIDAI',
      debugShowCheckedModeBanner: false,
      theme: YS.theme,
      routerConfig: appRouter,
    );
  }
}
