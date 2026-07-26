import 'package:flutter_test/flutter_test.dart';

import 'package:leaf_scan_app/main.dart';

void main() {
  testWidgets('Home screen shows the app title', (WidgetTester tester) async {
    await tester.pumpWidget(const LeafScanApp());
    await tester.pump();

    expect(find.text('AgroScan'), findsOneWidget);
    expect(find.text('Smart Drone Pesticide Spraying'), findsOneWidget);
  });
}
