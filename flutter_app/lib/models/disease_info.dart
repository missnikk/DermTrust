class DiseaseInfo {
  final String id;
  final String displayName;
  final String category;
  final String description;
  final List<String> visualHallmarks;
  final List<String> clinicallyRecognizedSubtypes;
  final String firstLineAdvice;
  final List<String> redFlags;

  const DiseaseInfo({
    required this.id,
    required this.displayName,
    required this.category,
    required this.description,
    required this.visualHallmarks,
    required this.clinicallyRecognizedSubtypes,
    required this.firstLineAdvice,
    required this.redFlags,
  });

  static const Map<String, DiseaseInfo> registry = {
    'acne': DiseaseInfo(
      id: 'acne',
      displayName: 'Acne',
      category: 'Inflammatory Pilosebaceous Disorder',
      description:
          'A multifactorial inflammatory disease of the pilosebaceous units characterized by comedones, papules, pustules, and sometimes nodules or cysts.',
      visualHallmarks: [
        'Open comedones (blackheads) & closed comedones (whiteheads)',
        'Erythematous inflammatory papules and pustules',
        'Centrofacial, forehead, and jawline distribution',
        'Potential post-inflammatory erythema or hyperpigmentation',
      ],
      clinicallyRecognizedSubtypes: [
        'Comedonal Acne (Mild, non-inflammatory)',
        'Papulopustular Acne (Moderate inflammatory)',
        'Nodulocystic Acne (Severe deep inflammatory)',
      ],
      firstLineAdvice:
          'Maintain gentle skin hygiene using non-comedogenic cleansers. Avoid picking, squeezing, or aggressive scrubbing, which exacerbates scarring.',
      redFlags: [
        'Deep painful cystic nodules',
        'Widespread scarring or psychological distress',
        'Sudden severe eruption accompanied by fever',
      ],
    ),
    'melasma': DiseaseInfo(
      id: 'melasma',
      displayName: 'Melasma',
      category: 'Acquired Hyperpigmentation Disorder',
      description:
          'A chronic, acquired dermatosis characterized by symmetric, irregular light-to-dark brown macules and patches in sun-exposed facial areas.',
      visualHallmarks: [
        'Irregular, reticulated brown to brownish-gray patches',
        'Malar (cheeks), frontal (forehead), and mandibular distribution',
        'Absence of erythema, induration, or epidermal scaling',
        'Exacerbation following UV radiation and hormonal triggers',
      ],
      clinicallyRecognizedSubtypes: [
        'Centrofacial Pattern (Forehead, nose, chin, upper lip)',
        'Malar Pattern (Cheeks and nose)',
        'Mandibular Pattern (Jawline)',
      ],
      firstLineAdvice:
          'Broad-spectrum UV protection (SPF 50+ with tinted zinc oxide / iron oxide) is critical. Wear wide-brimmed hats and avoid direct sunlight.',
      redFlags: [
        'Rapid asymmetric lesion enlargement',
        'Lesions exhibiting variegated black/blue colors or ulceration',
        'Suspicion of melanoma or lentigo maligna',
      ],
    ),
    'psoriasis': DiseaseInfo(
      id: 'psoriasis',
      displayName: 'Psoriasis',
      category: 'Chronic Immune-Mediated Inflammatory Skin Disorder',
      description:
          'A systemic immune-mediated condition with cutaneous manifestation of sharply demarcated erythematous plaques covered by silvery-white micaceous scales.',
      visualHallmarks: [
        'Well-demarcated salmon-pink to erythematous plaques',
        'Thick, micaceous silvery-white scaling (Auspitz sign)',
        'Extensor surfaces (elbows, knees, scalp, lumbosacral region)',
        'Koebner phenomenon (lesions appearing at sites of trauma)',
      ],
      clinicallyRecognizedSubtypes: [
        'Plaque Psoriasis (Psoriasis vulgaris - 85-90% of cases)',
        'Guttate Psoriasis (Drop-like lesions following streptococcal infection)',
        'Inverse Psoriasis (Intertriginous smooth flexural lesions)',
      ],
      firstLineAdvice:
          'Apply emollient ointments liberally to preserve skin barrier hydration and soften thick scales. Avoid skin trauma and emotional stress.',
      redFlags: [
        'Generalized body erythroderma (>90% skin involvement)',
        'Pustular eruptions accompanied by systemic fever',
        'Severe joint pain, stiffness, or dactylitis (Psoriatic Arthritis)',
      ],
    ),
    'ringworm': DiseaseInfo(
      id: 'ringworm',
      displayName: 'Ringworm',
      category: 'Superficial Dermatophyte Fungal Infection',
      description:
          'A contagious superficial fungal infection of glabrous skin caused by dermatophytes (Trichophyton, Microsporum, or Epidermophyton species).',
      visualHallmarks: [
        'Annular circular or polycyclic erythematous plaques',
        'Raised, active, scaly advancing border',
        'Characteristic central clearing (ring-like appearance)',
        'Pruritus (itching) that intensifies in warm/humid environments',
      ],
      clinicallyRecognizedSubtypes: [
        'Tinea Corporis (Glabrous body skin)',
        'Tinea Faciei (Facial non-bearded skin)',
        'Tinea Cruris (Groin and perineum - jock itch)',
      ],
      firstLineAdvice:
          'Maintain dry, clean skin. Do not share towels, combs, or personal clothing. Avoid applying topical steroid creams, which mask fungal features (Tinea incognito).',
      redFlags: [
        'Secondary bacterial infection with purulent drainage',
        'Immune-compromised status with extensive spreading',
        'Involvement of scalp with boggy swelling (Kerion)',
      ],
    ),
    'vitiligo': DiseaseInfo(
      id: 'vitiligo',
      displayName: 'Vitiligo',
      category: 'Autoimmune Depigmentary Disorder',
      description:
          'An acquired chronic autoimmune condition where melanocytes are selectively destroyed, resulting in stark, chalky-white macules and patches.',
      visualHallmarks: [
        'Complete depigmentation (chalky, enamel-white macules)',
        'Sharp, distinct anatomical borders with normal adjacent skin',
        'Symmetric acrofacial (hands, lips, eyes) or segmental patterns',
        'Leukotrichia (loss of pigment in hair within the lesion)',
      ],
      clinicallyRecognizedSubtypes: [
        'Non-Segmental Vitiligo (Generalized, bilateral symmetric - most common)',
        'Segmental Vitiligo (Unilateral dermatomal distribution, rapid onset)',
        'Focal / Mucosal Vitiligo (Limited isolated lesions)',
      ],
      firstLineAdvice:
          'Strict sun protection is vital because depigmented skin lacks protective melanin and burns rapidly. Consult a dermatologist for phototherapy evaluation.',
      redFlags: [
        'Rapid progression across large body surface areas',
        'Severe sunburn or blister formation on depigmented skin',
        'Associated symptoms of autoimmune thyroiditis or pernicious anemia',
      ],
    ),
  };

  static DiseaseInfo? getById(String id) {
    return registry[id.toLowerCase().trim()];
  }
}
