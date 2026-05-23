import 'package:go_router/go_router.dart';
import 'screens/home_screen.dart';
import 'screens/enroll_screen.dart';
import 'screens/authenticate_screen.dart';
import 'screens/pipeline_screen.dart';
import 'screens/screens.dart';

final appRouter = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(path: '/',             builder: (_, __) => const HomeScreen()),
    GoRoute(path: '/enroll',       builder: (_, __) => const EnrollScreen()),
    GoRoute(path: '/authenticate', builder: (_, __) => const AuthenticateScreen()),
    GoRoute(path: '/verify',       builder: (_, __) => const VerifyScreen()),
    GoRoute(path: '/pipeline',     builder: (_, __) => const PipelineScreen()),
    GoRoute(path: '/history',      builder: (_, __) => const HistoryScreen()),
    GoRoute(path: '/settings',     builder: (_, __) => const SettingsScreen()),
  ],
);
