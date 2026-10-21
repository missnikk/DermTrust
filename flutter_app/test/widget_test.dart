import 'package:flutter_test/flutter_test.dart';
import 'package:dermtrust/main.dart';
import 'package:dermtrust/utils/constants.dart';

void main() {
  testWidgets('DermTrustApp launches and displays branding', (WidgetTester tester) async {
    await tester.pumpWidget(const DermTrustApp());
    expect(find.text(AppConstants.appName), findsOneWidget);
  });
}
